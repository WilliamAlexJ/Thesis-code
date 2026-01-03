install.packages(c("tseries", "forecast", "FinTS"))

library(tseries)
library(forecast)
library(FinTS)

data <- read.csv("AAPL_log_returns_2010_to_2024.csv")

returns <- as.numeric(data$log_return)

returns <- returns[!is.na(returns)]


adf_result <- adf.test(returns)
print(adf_result)

lb_returns <- Box.test(returns, lag = 20, type = "Ljung-Box")
print(lb_returns)

lb_sq_returns <- Box.test(returns^2, lag = 20, type = "Ljung-Box")
print(lb_sq_returns)

arch_test <- ArchTest(returns, lags = 20)
print(arch_test)

jb <- jarque.bera.test(returns)
print(jb)
