"""
Task 3 runner for the Milan internet traffic forecast.

Three models are fit independently for three grid squares:
  SARIMA  one statistical baseline with daily seasonality
  LSTM    two layer recurrent network with a 144 step window
  TCN     dilated causal convolutions, same window as LSTM

Each fit saves its prediction parquet, model artifact, and forecast plot the
moment it finishes, so a crash in the middle of the run never wipes everything.

Designed for an 8 GB M3 MacBook Air. We work one model at a time, free memory
between fits, and never hold more than one model in RAM. Run from the project
root as:

    python src/task3_forecast.py --model all
    python src/task3_forecast.py --model sarima
    python src/task3_forecast.py --model lstm --areas A_5161,B_4159
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never pop a window; we are headless here
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# Project paths. Resolved relative to this file so the script works from any cwd.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T1_DIR = ROOT / "processed" / "task1"
T3_DIR = ROOT / "processed" / "task3"
SCALER_DIR = T3_DIR / "scalers"
MODEL_DIR = T3_DIR / "models"
PRED_DIR = T3_DIR / "predictions"
PLOT_DIR = T3_DIR / "plots"
TABLE_DIR = T3_DIR / "tables"
LOG_DIR = T3_DIR / "logs"
for d in (MODEL_DIR, PRED_DIR, PLOT_DIR, TABLE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Three target areas. A_5161 is the highest total traffic square from Task 1.
# B and C are the two squares specified in the brief.
AREAS = {
    "A_5161": 5161,
    "B_4159": 4159,
    "C_4556": 4556,
}

# Time boundaries. Everything is in Europe/Rome to keep midnight aligned with
# the city the data came from.
TZ = "Europe/Rome"
TRAIN_START = pd.Timestamp("2013-11-01", tz=TZ)
TEST_START = pd.Timestamp("2013-12-16", tz=TZ)
TEST_END = pd.Timestamp("2013-12-23", tz=TZ)  # exclusive
VAL_DAYS = 15            # last 15 days of training are the NN validation tail
WINDOW = 144             # one day of ten minute bins. Used for NN input.
SEASONAL_M = 144         # daily seasonality for SARIMA

# SARIMA order. Single fit per area, no grid search. If this proves too slow
# on the first area we fall back to ARIMA + seasonal naive (handled in main).
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEAS_ORDER = (1, 1, 1, SEASONAL_M)
SARIMA_MAXITER = 50

# NN training config
EPOCHS = 20
BATCH = 64
LR = 1e-3
PATIENCE = 5

SEED = 42


# ---------------------------------------------------------------------------
# Tiny logger that also tees to processed/task3/logs/run.log
# ---------------------------------------------------------------------------

class TeeLog:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()


log = TeeLog(LOG_DIR / "run.log")


# ---------------------------------------------------------------------------
# Data loading. Builds the three series as 10 minute regular pandas Series.
# ---------------------------------------------------------------------------

def load_series(square_id: int) -> pd.Series:
    """Load one square as a regular 10 minute series in Europe/Rome.

    The parquet was written by the Task 1 pipeline with a UTC DatetimeIndex
    named timestamp and a single float32 column internet_activity. We convert
    to Europe/Rome, drop dupes, and reindex onto the full 10 minute grid.
    """
    fp = T1_DIR / f"square_{square_id}.parquet"
    df = pd.read_parquet(fp)
    s = df["internet_activity"].astype(np.float64)
    # Localize to Europe/Rome so midnight aligns with the city
    s.index = s.index.tz_convert(TZ)
    s = s.sort_index()
    s = s.groupby(s.index).max()  # cheap insurance against duplicates
    full = pd.date_range(TRAIN_START, TEST_END, freq="10min", tz=TZ, inclusive="left")
    s = s.reindex(full).interpolate(limit=6).fillna(method="bfill").fillna(0)
    s.name = f"square_{square_id}"
    return s


def split_series(s: pd.Series):
    """Split into train (with NN validation tail) and test windows."""
    train = s.loc[:TEST_START - pd.Timedelta("10min")]
    test = s.loc[TEST_START:TEST_END - pd.Timedelta("10min")]
    val_cut = TEST_START - pd.Timedelta(days=VAL_DAYS)
    nn_train = train.loc[:val_cut - pd.Timedelta("10min")]
    nn_val = train.loc[val_cut:]
    return train, nn_train, nn_val, test


def load_scaler(label: str):
    with open(SCALER_DIR / f"{label}.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Metrics. MAPE has a floor of 1 in the denominator so quiet night hours
# (traffic close to zero) do not blow up the percentage.
# ---------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ---------------------------------------------------------------------------
# Plot helpers. One forecast plot per (model, area).
# ---------------------------------------------------------------------------

def shade_weekends(ax, start: pd.Timestamp, end: pd.Timestamp):
    cur = start.normalize()
    while cur <= end:
        if cur.weekday() >= 5:  # Saturday or Sunday
            ax.axvspan(cur, cur + pd.Timedelta(days=1), color="0.92", zorder=0)
        cur += pd.Timedelta(days=1)


def save_forecast_plot(label: str, model_name: str, y_true: pd.Series, y_pred: pd.Series, m: dict):
    fig, ax = plt.subplots(figsize=(12, 4))
    shade_weekends(ax, y_true.index[0], y_true.index[-1])
    ax.plot(y_true.index, y_true.values, color="#1f77b4", lw=1.4, label="actual")
    ax.plot(y_pred.index, y_pred.values, color="#ff7f0e", lw=1.2, ls="--", label="predicted")
    ax.set_title(
        f"{model_name} forecast, area {label} "
        f"(MAE={m['MAE']:.1f}, RMSE={m['RMSE']:.1f}, MAPE={m['MAPE']:.1f}%)"
    )
    ax.set_xlabel("date")
    ax.set_ylabel("internet activity (CDR units)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = PLOT_DIR / f"forecast_{model_name.lower()}_{label}.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    return fp


def save_prediction(label: str, model_name: str, y_true: pd.Series, y_pred: pd.Series):
    df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred.values}, index=y_true.index)
    df.index.name = "timestamp"
    fp = PRED_DIR / f"{model_name.lower()}_{label}.parquet"
    df.to_parquet(fp)
    return fp


# ---------------------------------------------------------------------------
# Model 1. SARIMA
# ---------------------------------------------------------------------------

def fit_sarima_for_area(label: str, train: pd.Series, test: pd.Series, fallback: bool):
    """Fit one SARIMAX per area. On fallback we use ARIMA plus a 144 step
    seasonal naive residual so we still get a working statistical baseline
    in seconds instead of minutes."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    log(f"  SARIMA {label} fitting (n_train={len(train)}) ...")
    t0 = time.perf_counter()
    if fallback:
        # ARIMA without seasonal terms plus a 144 step seasonal naive residual.
        # The seasonal naive part captures the daily cycle directly from the
        # training tail without paying the SARIMAX seasonal cost.
        m_arima = SARIMAX(
            train.values,
            order=(2, 1, 2),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=True,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=80, method="lbfgs")
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        arima_fc = m_arima.get_forecast(len(test)).predicted_mean
        # Seasonal naive: y_pred_seas[t] = train[-SEASONAL_M + (t % SEASONAL_M)]
        seas = train.values[-SEASONAL_M:]
        naive = np.array([seas[i % SEASONAL_M] for i in range(len(test))])
        # Blend: half ARIMA forecast (captures slow trend), half seasonal naive
        # (captures the daily cycle). Both halves on original scale.
        pred = 0.5 * arima_fc + 0.5 * naive
        infer_s = time.perf_counter() - t1
        # Save the smaller ARIMA fit
        with open(MODEL_DIR / f"sarima_{label}.pkl", "wb") as fh:
            pickle.dump({"kind": "ARIMA+seasonal_naive", "arima_params": m_arima.params}, fh)
    else:
        m = SARIMAX(
            train.values,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEAS_ORDER,
            simple_differencing=True,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=SARIMA_MAXITER, method="lbfgs")
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = m.get_forecast(len(test)).predicted_mean
        infer_s = time.perf_counter() - t1
        with open(MODEL_DIR / f"sarima_{label}.pkl", "wb") as fh:
            pickle.dump({"kind": "SARIMA", "params": m.params}, fh)
    # Traffic is non negative, clip floor at zero
    pred = np.clip(pred, 0, None)
    y_pred = pd.Series(pred, index=test.index)
    return y_pred, train_s, infer_s


# ---------------------------------------------------------------------------
# Models 2 and 3. NN training loop shared between LSTM and TCN.
# ---------------------------------------------------------------------------

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(SEED)
    except Exception:
        pass


def make_windows(series_scaled: np.ndarray, L: int):
    """Sliding windows. Returns X of shape (N, L, 1) and y of shape (N,)."""
    arr = series_scaled.astype(np.float32)
    if len(arr) <= L:
        return np.empty((0, L, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X = np.lib.stride_tricks.sliding_window_view(arr[:-1], L)
    X = X[:len(arr) - L]
    y = arr[L:]
    return X[..., None], y


def build_lstm():
    import torch
    from torch import nn

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                                dropout=0.2, batch_first=True)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    return LSTMRegressor()


def build_tcn():
    import torch
    from torch import nn
    from torch.nn.utils import weight_norm

    class Chomp1d(nn.Module):
        def __init__(self, chomp): super().__init__(); self.chomp = chomp
        def forward(self, x): return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x

    class Block(nn.Module):
        def __init__(self, in_c, out_c, k, d):
            super().__init__()
            pad = (k - 1) * d
            self.conv1 = weight_norm(nn.Conv1d(in_c, out_c, k, padding=pad, dilation=d))
            self.chomp1 = Chomp1d(pad)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.2)
            self.conv2 = weight_norm(nn.Conv1d(out_c, out_c, k, padding=pad, dilation=d))
            self.chomp2 = Chomp1d(pad)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(0.2)
            self.down = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
            self.act = nn.ReLU()

        def forward(self, x):
            y = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
            y = self.drop2(self.relu2(self.chomp2(self.conv2(y))))
            res = x if self.down is None else self.down(x)
            return self.act(y + res)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            chans = [1, 32, 32, 32]
            dilations = [1, 2, 4]
            blocks = []
            for i in range(3):
                blocks.append(Block(chans[i], chans[i + 1], 3, dilations[i]))
            self.net = nn.Sequential(*blocks)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            # x is (B, L, 1). Conv1d needs (B, 1, L).
            y = self.net(x.transpose(1, 2))
            return self.head(y[:, :, -1]).squeeze(-1)

    return TCN()


def train_nn(model, X_train, y_train, X_val, y_val, device, model_name: str, label: str):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = torch.nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val = float("inf")
    best_state = None
    bad = 0
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        tr_loss = tot / max(n, 1)

        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                tot += loss_fn(pred, yb).item() * len(yb)
                n += len(yb)
        val_loss = tot / max(n, 1)
        sched.step(val_loss)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        log(f"    {model_name} {label} epoch {ep:02d}/{EPOCHS} train={tr_loss:.5f} val={val_loss:.5f}{' *' if bad == 0 else ''}")
        if bad >= PATIENCE:
            log(f"    {model_name} {label} early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist epoch history and weights
    pd.DataFrame(hist).to_csv(LOG_DIR / f"{model_name.lower()}_{label}_history.csv", index=False)
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name.lower()}_{label}.pt")
    return model


def rolling_predict(model, history: np.ndarray, n_steps: int, L: int, device):
    """Rolling one step ahead inference on scaled values. At step t we feed
    the most recent L true values into the model. The true values become
    available because this is the evaluation phase; the model never sees
    the next step before predicting it."""
    import torch
    model.eval()
    preds = np.empty(n_steps, dtype=np.float32)
    with torch.no_grad():
        for t in range(n_steps):
            window = history[t : t + L]
            x = torch.from_numpy(window[None, :, None].astype(np.float32)).to(device)
            preds[t] = float(model(x).cpu().numpy())
    return preds


def fit_nn_for_area(label: str, model_name: str, model_factory, train: pd.Series,
                    nn_train: pd.Series, nn_val: pd.Series, test: pd.Series):
    import torch
    set_seed()

    scaler = load_scaler(label)
    # Scaled arrays
    full_scaled = scaler.transform(
        np.concatenate([train.values, test.values]).reshape(-1, 1)
    ).ravel().astype(np.float32)
    train_n = len(train)
    nn_train_n = len(nn_train)

    X_tr, y_tr = make_windows(full_scaled[:nn_train_n], WINDOW)
    X_val, y_val = make_windows(full_scaled[nn_train_n - WINDOW: train_n], WINDOW)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"  {model_name} {label} train_windows={len(X_tr)} val_windows={len(X_val)} device={device}")

    t0 = time.perf_counter()
    model = model_factory()
    model = train_nn(model, X_tr, y_tr, X_val, y_val, device, model_name, label)
    train_s = time.perf_counter() - t0

    # Rolling one step ahead on the test region. History context is the last
    # WINDOW values of training plus all of test except the last point.
    context = full_scaled[train_n - WINDOW: train_n + len(test) - 1]
    t1 = time.perf_counter()
    preds_scaled = rolling_predict(model, context, len(test), WINDOW, device)
    infer_s = time.perf_counter() - t1

    # Back to original scale, clip non negative
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    preds = np.clip(preds, 0, None)
    y_pred = pd.Series(preds, index=test.index)

    # Free the model immediately so the next area starts clean
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return y_pred, train_s, infer_s


# ---------------------------------------------------------------------------
# Main loop. Runs one model across the selected areas, saving as it goes.
# ---------------------------------------------------------------------------

def run_sarima(areas: list, fallback: bool):
    results = {}
    for label in areas:
        sq = AREAS[label]
        log(f"SARIMA on {label} (sq {sq})")
        s = load_series(sq)
        train, _, _, test = split_series(s)
        y_pred, train_s, infer_s = fit_sarima_for_area(label, train, test, fallback)
        m = metrics(test.values, y_pred.values)
        save_prediction(label, "SARIMA", test, y_pred)
        save_forecast_plot(label, "SARIMA", test, y_pred, m)
        results[label] = {
            "metrics": m,
            "train_s": train_s,
            "infer_s": infer_s,
            "n_test": len(test),
        }
        log(f"  done. train={train_s:.1f}s infer={infer_s:.2f}s "
            f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")
        del s, train, test, y_pred
        gc.collect()
    update_metrics_json("SARIMA", results, fallback=fallback)


def run_nn(model_name: str, areas: list):
    factory = build_lstm if model_name == "LSTM" else build_tcn
    results = {}
    for label in areas:
        sq = AREAS[label]
        log(f"{model_name} on {label} (sq {sq})")
        s = load_series(sq)
        train, nn_train, nn_val, test = split_series(s)
        y_pred, train_s, infer_s = fit_nn_for_area(
            label, model_name, factory, train, nn_train, nn_val, test
        )
        m = metrics(test.values, y_pred.values)
        save_prediction(label, model_name, test, y_pred)
        save_forecast_plot(label, model_name, test, y_pred, m)
        results[label] = {
            "metrics": m,
            "train_s": train_s,
            "infer_s": infer_s,
            "n_test": len(test),
        }
        log(f"  done. train={train_s:.1f}s infer={infer_s:.2f}s "
            f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")
        del s, train, nn_train, nn_val, test, y_pred
        gc.collect()
    update_metrics_json(model_name, results)


def update_metrics_json(model_name: str, results: dict, fallback: bool = False):
    """Merge this model's results into the running metrics.json blob."""
    path = T3_DIR / "metrics.json"
    blob = {}
    if path.exists():
        blob = json.loads(path.read_text())
    blob.setdefault("models", {})
    entry = {"results": results}
    if fallback and model_name == "SARIMA":
        entry["variant"] = "ARIMA(2,1,2) + 144 step seasonal naive blend"
    else:
        entry["variant"] = {
            "SARIMA": f"SARIMAX{SARIMA_ORDER}{SARIMA_SEAS_ORDER}",
            "LSTM": "2 layer hidden 64 dropout 0.2",
            "TCN": "3 dilated blocks (1,2,4), 32 channels, kernel 3",
        }[model_name]
    blob["models"][model_name] = entry
    blob["env"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    path.write_text(json.dumps(blob, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "sarima", "lstm", "tcn"])
    parser.add_argument("--areas", default=",".join(AREAS.keys()),
                        help="comma separated list, default all three")
    parser.add_argument("--sarima-fallback", action="store_true",
                        help="skip the daily seasonality SARIMA and use ARIMA + seasonal naive")
    args = parser.parse_args()
    areas = [a.strip() for a in args.areas.split(",") if a.strip()]
    unknown = [a for a in areas if a not in AREAS]
    if unknown:
        log("unknown areas:", unknown)
        sys.exit(1)

    log("=" * 60)
    log(f"task3_forecast starting model={args.model} areas={areas} fallback={args.sarima_fallback}")
    log(f"python {sys.version.split()[0]} on {platform.platform()}")
    log("=" * 60)

    t_total = time.perf_counter()
    if args.model in ("all", "sarima"):
        run_sarima(areas, fallback=args.sarima_fallback)
    if args.model in ("all", "lstm"):
        run_nn("LSTM", areas)
    if args.model in ("all", "tcn"):
        run_nn("TCN", areas)
    log(f"total wall time {time.perf_counter() - t_total:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
