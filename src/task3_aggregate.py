"""
Reads the saved per (model, area) prediction parquets and produces the rubric
required tables. Three metric tables (one per area), one timing summary, one
failure analysis plot and consolidated metrics.json. Nothing heavy here, just
file I/O and small dataframes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T3 = ROOT / "processed" / "task3"
PRED = T3 / "predictions"
TABLE = T3 / "tables"
PLOT = T3 / "plots"
TABLE.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)

AREAS = ["A_5161", "B_4159", "C_4556"]
MODELS = ["SARIMA", "LSTM", "TCN"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def load_pred(model: str, label: str) -> pd.DataFrame | None:
    fp = PRED / f"{model.lower()}_{label}.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


def write_md_table(rows, header, path: Path):
    """Plain markdown table writer with the best per row metric bolded."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [str(r[h]) for h in header]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def make_metric_tables():
    """Three tables. One per area. Rows are SARIMA, LSTM, TCN."""
    summary = {}
    for label in AREAS:
        rows = []
        for model in MODELS:
            df = load_pred(model, label)
            if df is None:
                rows.append({"Model": model, "MAE": "n/a", "RMSE": "n/a", "MAPE (%)": "n/a"})
                continue
            m = metrics(df["y_true"].values, df["y_pred"].values)
            rows.append({
                "Model": model,
                "MAE": f"{m['MAE']:.2f}",
                "RMSE": f"{m['RMSE']:.2f}",
                "MAPE (%)": f"{m['MAPE']:.2f}",
            })
            summary.setdefault(label, {})[model] = m
        # CSV
        pd.DataFrame(rows).to_csv(TABLE / f"metrics_{label}.csv", index=False)
        # Markdown
        write_md_table(rows, ["Model", "MAE", "RMSE", "MAPE (%)"],
                       TABLE / f"metrics_{label}.md")
    return summary


def make_timing_table(metrics_json: dict):
    """Pull train and infer times from metrics.json and tabulate."""
    rows = []
    avg = {m: [] for m in MODELS}
    for model in MODELS:
        entry = metrics_json.get("models", {}).get(model)
        if not entry:
            continue
        for label, res in entry["results"].items():
            rows.append({
                "Area": label,
                "Model": model,
                "Train (s)": f"{res['train_s']:.2f}",
                "Inference (s)": f"{res['infer_s']:.3f}",
                "Steps per sec": f"{res['n_test'] / max(res['infer_s'], 1e-6):.1f}",
            })
            avg[model].append(res["train_s"])
    pd.DataFrame(rows).to_csv(TABLE / "timing_summary.csv", index=False)
    write_md_table(rows, ["Area", "Model", "Train (s)", "Inference (s)", "Steps per sec"],
                   TABLE / "timing_summary.md")

    avg_rows = []
    for model in MODELS:
        if avg[model]:
            avg_rows.append({"Model": model, "Mean train (s)": f"{np.mean(avg[model]):.2f}",
                             "Areas counted": len(avg[model])})
    pd.DataFrame(avg_rows).to_csv(TABLE / "timing_avg_per_model.csv", index=False)
    write_md_table(avg_rows, ["Model", "Mean train (s)", "Areas counted"],
                   TABLE / "timing_avg_per_model.md")


def shade_weekends(ax, start, end):
    cur = start.normalize()
    while cur <= end:
        if cur.weekday() >= 5:
            ax.axvspan(cur, cur + pd.Timedelta(days=1), color="0.92", zorder=0)
        cur += pd.Timedelta(days=1)


def failure_analysis():
    """Slide a 48 hour window across the test period for every (model, area)
    pair, compute rolling RMSE, and pick the single worst window. Plot all
    three models against the actual over that window."""
    WIN = 288  # 48 hours of ten minute bins

    # Load every prediction
    cache = {}
    for label in AREAS:
        for model in MODELS:
            df = load_pred(model, label)
            if df is not None:
                cache[(label, model)] = df

    # Compute worst window per (label, model)
    worst = None
    for (label, model), df in cache.items():
        err = (df["y_true"].values - df["y_pred"].values) ** 2
        n = len(err)
        if n < WIN:
            continue
        # Rolling sum of squared errors
        cum = np.concatenate([[0.0], np.cumsum(err)])
        rmse = np.sqrt((cum[WIN:] - cum[:-WIN]) / WIN)
        idx = int(np.argmax(rmse))
        peak = float(rmse[idx])
        if worst is None or peak > worst["rmse"]:
            worst = {"label": label, "model": model, "start": idx, "rmse": peak}

    if worst is None:
        return None

    label = worst["label"]
    start = worst["start"]
    base_df = cache[(label, MODELS[0])] if (label, MODELS[0]) in cache else next(
        df for (l, m), df in cache.items() if l == label
    )
    window_index = base_df.index[start: start + WIN]

    fig, ax = plt.subplots(figsize=(12, 5))
    shade_weekends(ax, window_index[0], window_index[-1])
    ax.plot(window_index, base_df["y_true"].iloc[start: start + WIN], color="black",
            lw=1.5, label="actual")
    colors = {"SARIMA": "#1f77b4", "LSTM": "#2ca02c", "TCN": "#d62728"}
    for model in MODELS:
        df = cache.get((label, model))
        if df is None:
            continue
        ax.plot(window_index, df["y_pred"].iloc[start: start + WIN],
                color=colors[model], lw=1.1, ls="--", label=model)
    ax.set_title(
        f"Worst 48 hour window across all models. Area {label}. "
        f"Peak RMSE driver was {worst['model']} at RMSE={worst['rmse']:.1f}"
    )
    ax.set_xlabel("date")
    ax.set_ylabel("internet activity (CDR units)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT / "failure_analysis.png", dpi=140)
    plt.close(fig)
    return worst


def main():
    metrics_json_path = T3 / "metrics.json"
    blob = {}
    if metrics_json_path.exists():
        blob = json.loads(metrics_json_path.read_text())

    summary = make_metric_tables()
    blob["metrics_summary"] = summary
    make_timing_table(blob)
    worst = failure_analysis()
    if worst is not None:
        blob["worst_window"] = {
            "area": worst["label"],
            "model": worst["model"],
            "window_start_index": worst["start"],
            "rmse": worst["rmse"],
        }
    metrics_json_path.write_text(json.dumps(blob, indent=2))
    print("aggregation complete")
    print("worst window:", worst)


if __name__ == "__main__":
    main()
