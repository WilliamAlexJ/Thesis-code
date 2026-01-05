library(rugarch)
library(xts)
library(tensorflow)
library(keras3)

#LOAD DATA
df <- read.csv("NVDA_log_returns_2010_to_2024.csv")

df$date <- as.Date(df$date)

log_returns <- as.numeric(df$log_return)
dates       <- df$date
n_total     <- length(log_returns)

window_size   <- 1000
retrain_every <- 50
batch_size    <- 32
alpha_vec     <- c(0.01, 0.05)

#LSTM configurations
lstm_configs <- list(
  list(name = "LSTM_lb30_e20", lookback = 30, units = 32, epochs = 20),
  list(name = "LSTM_lb30_e30", lookback = 30, units = 32, epochs = 30),
  list(name = "LSTM_lb60_e20", lookback = 60, units = 32, epochs = 20),
  list(name = "LSTM_lb60_e30", lookback = 60, units = 32, epochs = 30)
)

set.seed(123)
tensorflow::tf$random$set_seed(123L)

#FUNCTIONS

tick_loss <- function(VaR, ret, alpha) {
  ind <- ifelse(VaR >= ret, 1, 0)
  (ind - alpha) * (VaR - ret)
}

kupiec_test <- function(ret, VaR, alpha) {
  I <- ifelse(ret < VaR, 1, 0)
  Tn <- length(I)
  N  <- sum(I)
  p_hat <- N / Tn
  
  L0 <- (1 - alpha)^(Tn - N) * alpha^N
  L1 <- (1 - p_hat)^(Tn - N) * p_hat^N
  
  LR_uc <- -2 * log(L0 / L1)
  p_value <- 1 - pchisq(LR_uc, df = 1)
  
  list(
    N_exceed = N,
    T        = Tn,
    alpha    = alpha,
    LR_uc    = LR_uc,
    p_value  = p_value
  )
}

christoffersen_tests <- function(ret, VaR, alpha) {
  I <- ifelse(ret < VaR, 1, 0)
  I_lag <- I[-length(I)]
  I_now <- I[-1]
  
  N00 <- sum(I_lag == 0 & I_now == 0)
  N01 <- sum(I_lag == 0 & I_now == 1)
  N10 <- sum(I_lag == 1 & I_now == 0)
  N11 <- sum(I_lag == 1 & I_now == 1)
  
  pi0 <- if ((N00 + N01) > 0) N01 / (N00 + N01) else 0
  pi1 <- if ((N10 + N11) > 0) N11 / (N10 + N11) else 0
  pi  <- (N01 + N11) / (N00 + N01 + N10 + N11)
  
  L0_ind <- (1 - pi)^(N00 + N10) * pi^(N01 + N11)
  L1_ind <- (1 - pi0)^N00 * pi0^N01 * (1 - pi1)^N10 * pi1^N11
  
  LR_ind <- -2 * log(L0_ind / L1_ind)
  p_ind  <- 1 - pchisq(LR_ind, df = 1)
  
  Tn <- length(I)
  N  <- sum(I)
  p_hat <- N / Tn
  L0_uc <- (1 - alpha)^(Tn - N) * alpha^N
  L1_uc <- (1 - p_hat)^(Tn - N) * p_hat^N
  LR_uc <- -2 * log(L0_uc / L1_uc)
  p_uc  <- 1 - pchisq(LR_uc, df = 1)
  
  LR_cc <- LR_uc + LR_ind
  p_cc  <- 1 - pchisq(LR_cc, df = 2)
  
  list(
    N00   = N00, N01 = N01, N10 = N10, N11 = N11,
    LR_uc = LR_uc, p_uc = p_uc,
    LR_ind = LR_ind, p_ind = p_ind,
    LR_cc  = LR_cc, p_cc = p_cc
  )
}

dm_test_tick <- function(loss1, loss2) {
  d  <- loss1 - loss2
  Tn <- length(d)
  d_bar <- mean(d)
  s2 <- var(d)
  
  DM_stat <- d_bar / sqrt(s2 / Tn)
  p_value <- 2 * (1 - pt(abs(DM_stat), df = Tn - 1))
  
  list(
    DM_stat  = DM_stat,
    p_value  = p_value,
    mean_diff = d_bar
  )
}

#EGARCH ROLLING VAR

egarch_spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

rolling_egarch_var <- function(log_returns, spec, window_size, alpha) {
  n <- length(log_returns)
  VaR_vec <- rep(NA, n)
  
  for (t in window_size:(n - 1)) {
    ret_window <- log_returns[(t - window_size + 1):t]
    
    fit <- ugarchfit(
      spec = spec,
      data = ret_window,
      solver = "hybrid",
      solver.control = list(trace = 0)
    )
    
    fc <- ugarchforecast(fit, n.ahead = 1)
    sigma_fc <- sigma(fc)[1]
    shape <- coef(fit)["shape"]
    
    VaR_t1 <- qdist("std",
                    p     = alpha,
                    mu    = 0,
                    sigma = sigma_fc,
                    shape = shape)
    
    VaR_vec[t + 1] <- VaR_t1
  }
  
  VaR_vec
}

#LSTM QUANTILE MODEL (ROLLING)

quantile_loss <- function(alpha) {
  a <- tensorflow::tf$constant(alpha, dtype = tensorflow::tf$float32)
  function(y_true, y_pred) {
    e <- tensorflow::tf$cast(y_true - y_pred, dtype = tensorflow::tf$float32)
    tensorflow::tf$reduce_mean(
      tensorflow::tf$maximum(a * e, (a - 1) * e)
    )
  }
}

build_lstm_model <- function(lookback, units) {
  keras3::keras_model_sequential() |>
    keras3::layer_lstm(units = units, input_shape = c(lookback, 1)) |>
    keras3::layer_dense(units = 1)
}

train_lstm_on_window <- function(ret_window,
                                 lookback,
                                 alpha,
                                 units,
                                 epochs,
                                 batch_size = 32) {
  mu  <- mean(ret_window)
  sdv <- sd(ret_window)
  scaled <- (ret_window - mu) / sdv
  
  X_list <- list()
  y_vec  <- c()
  
  for (t in lookback:(length(scaled) - 1)) {
    X_seq <- scaled[(t - lookback + 1):t]
    y_val <- ret_window[t + 1]
    X_list[[length(X_list) + 1]] <- X_seq
    y_vec <- c(y_vec, y_val)
  }
  
  X_train <- array(unlist(X_list),
                   dim = c(length(X_list), lookback, 1))
  
  model <- build_lstm_model(lookback, units)
  
  model |>
    keras3::compile(
      optimizer = "adam",
      loss      = quantile_loss(alpha)
    )
  
  model |>
    keras3::fit(
      x = X_train,
      y = y_vec,
      epochs = epochs,
      batch_size = batch_size,
      verbose = 0
    )
  
  list(
    model    = model,
    mu       = mu,
    sd       = sdv,
    lookback = lookback
  )
}

predict_lstm_one_step <- function(model_obj, log_returns, t_index) {
  model    <- model_obj$model
  mu       <- model_obj$mu
  sdv      <- model_obj$sd
  lookback <- model_obj$lookback
  
  idx_start <- t_index - lookback + 1
  idx_end   <- t_index
  
  if (idx_start < 1) return(NA)
  
  X_seq <- (log_returns[idx_start:idx_end] - mu) / sdv
  X_in  <- array(X_seq, dim = c(1, lookback, 1))
  
  as.numeric(model$predict(X_in, verbose = 0))
}

rolling_lstm_var <- function(log_returns,
                             window_size,
                             lookback,
                             alpha,
                             units,
                             epochs,
                             batch_size = 32,
                             retrain_every = 50,
                             cfg_name = "") {
  n <- length(log_returns)
  VaR_vec <- rep(NA, n)
  
  last_trained_t <- NA
  model_obj <- NULL
  
  for (t in window_size:(n - 1)) {
    if (is.na(last_trained_t) || (t - last_trained_t) >= retrain_every) {
      ret_window <- log_returns[(t - window_size + 1):t]
      model_obj <- train_lstm_on_window(
        ret_window = ret_window,
        lookback   = lookback,
        alpha      = alpha,
        units      = units,
        epochs     = epochs,
        batch_size = batch_size
      )
      last_trained_t <- t
    }
    
    VaR_vec[t + 1] <- predict_lstm_one_step(model_obj, log_returns, t)
  }
  
  VaR_vec
}

#MAIN LOOP: EGACH VS MULTI-LSTM

backtest_summary <- data.frame(
  model      = character(),
  config     = character(),
  alpha      = numeric(),
  N_exceed   = integer(),
  T          = integer(),
  coverage   = numeric(),
  mean_loss  = numeric(),
  kupiec_LR  = numeric(),
  kupiec_p   = numeric(),
  cc_LR      = numeric(),
  cc_p       = numeric(),
  stringsAsFactors = FALSE
)

dm_summary <- data.frame(
  alpha      = numeric(),
  model      = character(),
  config     = character(),
  DM_stat    = numeric(),
  DM_p_value = numeric(),
  mean_diff  = numeric(),
  stringsAsFactors = FALSE
)

start_oos <- window_size + 1

for (alpha in alpha_vec) {
  
  VaR_eg <- rolling_egarch_var(
    log_returns = log_returns,
    spec        = egarch_spec,
    window_size = window_size,
    alpha       = alpha
  )
  
  VaR_eg_oos <- VaR_eg[start_oos:n_total]
  ret_oos    <- log_returns[start_oos:n_total]
  
  idx_eg <- which(!is.na(VaR_eg_oos))
  VaR_eg_use <- VaR_eg_oos[idx_eg]
  ret_eg_use <- ret_oos[idx_eg]
  
  loss_eg_use <- tick_loss(VaR_eg_use, ret_eg_use, alpha)
  exceed_eg   <- ifelse(ret_eg_use < VaR_eg_use, 1, 0)
  
  coverage_eg  <- mean(exceed_eg)
  mean_loss_eg <- mean(loss_eg_use)
  
  kupiec_eg <- kupiec_test(ret_eg_use, VaR_eg_use, alpha)
  christ_eg <- christoffersen_tests(ret_eg_use, VaR_eg_use, alpha)
  
  backtest_summary <- rbind(
    backtest_summary,
    data.frame(
      model     = "EGARCH",
      config    = "baseline",
      alpha     = alpha,
      N_exceed  = kupiec_eg$N_exceed,
      T         = kupiec_eg$T,
      coverage  = coverage_eg,
      mean_loss = mean_loss_eg,
      kupiec_LR = kupiec_eg$LR_uc,
      kupiec_p  = kupiec_eg$p_value,
      cc_LR     = christ_eg$LR_cc,
      cc_p      = christ_eg$p_cc,
      stringsAsFactors = FALSE
    )
  )
  
  for (cfg in lstm_configs) {
    VaR_lstm <- rolling_lstm_var(
      log_returns   = log_returns,
      window_size   = window_size,
      lookback      = cfg$lookback,
      alpha         = alpha,
      units         = cfg$units,
      epochs        = cfg$epochs,
      batch_size    = batch_size,
      retrain_every = retrain_every,
      cfg_name      = cfg$name
    )
    
    VaR_lstm_oos <- VaR_lstm[start_oos:n_total]
    
    idx_lstm <- which(!is.na(VaR_lstm_oos))
    common_idx <- intersect(idx_eg, idx_lstm)
    
    VaR_eg_common   <- VaR_eg_oos[common_idx]
    VaR_lstm_common <- VaR_lstm_oos[common_idx]
    ret_common      <- ret_oos[common_idx]
    
    loss_eg_common   <- tick_loss(VaR_eg_common, ret_common, alpha)
    loss_lstm_common <- tick_loss(VaR_lstm_common, ret_common, alpha)
    
    exceed_lstm <- ifelse(ret_common < VaR_lstm_common, 1, 0)
    coverage_lstm  <- mean(exceed_lstm)
    mean_loss_lstm <- mean(loss_lstm_common)
    
    kupiec_lstm <- kupiec_test(ret_common, VaR_lstm_common, alpha)
    christ_lstm <- christoffersen_tests(ret_common, VaR_lstm_common, alpha)
    
    backtest_summary <- rbind(
      backtest_summary,
      data.frame(
        model     = "LSTM",
        config    = cfg$name,
        alpha     = alpha,
        N_exceed  = kupiec_lstm$N_exceed,
        T         = kupiec_lstm$T,
        coverage  = coverage_lstm,
        mean_loss = mean_loss_lstm,
        kupiec_LR = kupiec_lstm$LR_uc,
        kupiec_p  = kupiec_lstm$p_value,
        cc_LR     = christ_lstm$LR_cc,
        cc_p      = christ_lstm$p_cc,
        stringsAsFactors = FALSE
      )
    )
    
    dm_res <- dm_test_tick(loss_eg_common, loss_lstm_common)
    
    dm_summary <- rbind(
      dm_summary,
      data.frame(
        alpha      = alpha,
        model      = "LSTM",
        config     = cfg$name,
        DM_stat    = dm_res$DM_stat,
        DM_p_value = dm_res$p_value,
        mean_diff  = dm_res$mean_diff,
        stringsAsFactors = FALSE
      )
    )
  }
}

#RESULTs  TO CSV
write.csv(
  backtest_summary,
  "backtest_summary_egarch_vs_lstm_rolling_multiLSTM.csv",
  row.names = FALSE
)

write.csv(
  dm_summary,
  "dm_summary_egarch_vs_lstm_rolling_multiLSTM.csv",
  row.names = FALSE
)
