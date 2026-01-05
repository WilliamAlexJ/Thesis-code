import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

#LOAD DATA
df = pd.read_csv("NVDA_log_returns_2010_to_2024.csv")
df["date"] = pd.to_datetime(df["date"])
dates = df["date"].values
log_returns = df["log_return"].astype(float).values
n_total = len(log_returns)

#GLOBAL SETTINGS
window_size   = 1000
retrain_every = 50
alpha_vec     = [0.01, 0.05]

tcn_configs = [
    {"name": "TCN_short", "lookback": 30, "channels": 32, "levels": 5},
    {"name": "TCN_long",  "lookback": 60, "channels": 32, "levels": 5},
]

torch.manual_seed(123)
np.random.seed(123)
device = torch.device("cpu")

#MODEL
class QuantileLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self, y_pred, y_true):
        e = y_true - y_pred
        return torch.mean(torch.maximum(self.alpha * e, (self.alpha - 1) * e))

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.0):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, channels=32, levels=5, kernel_size=3):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(levels):
            layers.append(TemporalBlock(in_ch, channels, kernel_size, dilation=2**i, dropout=0.0))
            in_ch = channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)[:, :, -1]
        return self.fc(y)

#TRAINING
def build_training_sequences(ret_window, lookback):
    mu = ret_window.mean()
    sd = ret_window.std(ddof=1)
    scaled = (ret_window - mu) / sd

    X, y = [], []
    for t in range(lookback - 1, len(scaled) - 1):
        X.append(scaled[t - lookback + 1:t + 1])
        y.append(ret_window[t + 1])

    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32)
    return X, y, mu, sd

def train_model(model, X, y, alpha, epochs=20, lr=1e-3):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = QuantileLoss(alpha)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    return model

def predict_one_step(model, series, t, mu, sd, lookback):
    seq = series[t - lookback + 1:t + 1]
    x = ((seq - mu) / sd).astype(np.float32).reshape(1, lookback, 1)
    x = torch.from_numpy(x)

    model.eval()
    with torch.no_grad():
        return float(model(x).item())

#ROLLING FORECAST
def rolling_tcn_forecast(series, dates, window_size, alpha, lookback, channels, levels,
                         epochs, retrain_every, cfg_name=""):
    n = len(series)
    VaR = np.full(n, np.nan)

    last_trained_t = None
    model = None
    mu = sd = None

    for t in range(window_size, n - 1):
        if last_trained_t is None or (t - last_trained_t) >= retrain_every:
            window = series[t - window_size + 1:t + 1]
            X, y, mu, sd = build_training_sequences(window, lookback)

            model = TCN(channels=channels, levels=levels, kernel_size=3)
            model = train_model(model, X, y, alpha, epochs=epochs)
            last_trained_t = t
            print(f"[TCN:{cfg_name}] retrained at t={t}, alpha={alpha}")

        VaR[t + 1] = predict_one_step(model, series, t, mu, sd, lookback)

    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "return": series,
        "VaR": VaR,
        "alpha": alpha,
        "model": "TCN",
        "config": cfg_name
    })
    return out

#MAIN: RUN + EXPORT
all_full = []

for alpha in alpha_vec:
    for cfg in tcn_configs:
        name = cfg["name"]
        lookback = cfg["lookback"]
        channels = cfg["channels"]
        levels = cfg["levels"]

        df_pred = rolling_tcn_forecast(
            series=log_returns,
            dates=dates,
            window_size=window_size,
            alpha=alpha,
            lookback=lookback,
            channels=channels,
            levels=levels,
            epochs=20,
            retrain_every=retrain_every,
            cfg_name=name
        )

        df_pred.to_csv(f"predictions_{name}_alpha{alpha}.csv", index=False)
        all_full.append(df_pred)

all_full_df = pd.concat(all_full, ignore_index=True)
all_full_df.to_csv("predictions_TCN_ALL.csv", index=False)

print("\nExported:")
print(" - predictions_TCN_ALL.csv (full-length, includes NaNs)")
print(" - predictions_<config>_alpha<alpha>.csv (full-length, includes NaNs)")
