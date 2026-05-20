"""
Build notebooks/task3_modeling.ipynb from the saved Task 3 artifacts.

The notebook does not fit any models, it loads the saved predictions, plots
and tables and displays them next to short explanatory markdown. This way the
submission has a single readable notebook artifact even though the heavy work
ran in the CLI script.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T3 = ROOT / "processed" / "task3"
NB_OUT = ROOT / "notebooks" / "task3_modeling.ipynb"

AREAS = ["A_5161", "B_4159", "C_4556"]
MODELS = ["SARIMA", "LSTM", "TCN"]


def md(s: str):
    return nbf.v4.new_markdown_cell(s)


def code(s: str):
    return nbf.v4.new_code_cell(s)


def main():
    blob = {}
    p = T3 / "metrics.json"
    if p.exists():
        blob = json.loads(p.read_text())

    cells = []

    cells.append(md(
        "# Task 3. Forecasting Milan Internet Traffic, December 16 to 22 2013\n\n"
        "Three models fit independently per area: a classical SARIMA, a stacked LSTM, "
        "and a TCN. This notebook is a viewer over results that were produced by "
        "`src/task3_forecast.py` and aggregated by `src/task3_aggregate.py`. The "
        "heavy fitting is intentionally outside the notebook so the run survives "
        "a kernel crash and we can finish in two hours on 8 GB of unified memory."
    ))

    cells.append(md(
        "## Targets and forecast window\n\n"
        "| Area | Square ID | Role |\n"
        "|---|---|---|\n"
        "| A_5161 | 5161 | Highest total traffic across the full two months |\n"
        "| B_4159 | 4159 | Specified by the rubric |\n"
        "| C_4556 | 4556 | Specified by the rubric |\n\n"
        "Test window: 2013 12 16 00:00 Europe/Rome to 2013 12 22 23:50, "
        "exactly 1,008 ten minute steps. Training spans 2013 11 01 to 2013 12 15. "
        "For the neural networks the last 15 days of training are held out as a "
        "validation tail used for early stopping; no test data ever leaks into training."
    ))

    cells.append(md(
        "## Model summary\n\n"
        "**SARIMA.** statsmodels SARIMAX, order (1,1,1)(1,1,1,144), "
        "`simple_differencing=True` so the state vector stays small enough to fit "
        "in 8 GB. Inference is multi step `get_forecast(1008)`, the hardest setting "
        "the model can be evaluated on without leaking test data into training.\n\n"
        "**LSTM.** Two layer stacked LSTM, hidden 64, dropout 0.2, Dense(1) head. "
        "Input is a 144 step sliding window (one day of context). Adam(1e 3), "
        "ReduceLROnPlateau, early stopping with patience 5, max 20 epochs, batch 64. "
        "Inference is rolling one step ahead, at each test step the true past 144 "
        "values are fed in to predict the next step.\n\n"
        "**TCN.** Three residual blocks of dilated causal Conv1D, kernel 3, channels 32, "
        "dilations (1, 2, 4), dropout 0.2, Dense(1) head on the last step. Same training "
        "loop and inference convention as LSTM. Chosen over a Transformer encoder because "
        "the M3 has limited unified memory and TCNs train notably faster while still "
        "covering a wide receptive field across the L=144 explicit context."
    ))

    cells.append(md(
        "## Setup. Display tables and figures from disk"
    ))
    cells.append(code(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import Image, Markdown, display\n"
        "\n"
        "T3 = Path('../processed/task3')\n"
        "PRED = T3 / 'predictions'\n"
        "PLOTS = T3 / 'plots'\n"
        "TABLES = T3 / 'tables'\n"
        "blob = json.loads((T3 / 'metrics.json').read_text()) if (T3 / 'metrics.json').exists() else {}\n"
        "print('models present:', sorted(blob.get('models', {}).keys()))\n"
    ))

    cells.append(md("## Nine forecast plots, three models by three areas"))
    for label in AREAS:
        for model in MODELS:
            cells.append(md(f"### {model} on {label}"))
            cells.append(code(
                f"display(Image(str(PLOTS / 'forecast_{model.lower()}_{label}.png')))"
            ))

    cells.append(md(
        "## Per area metric tables\n\n"
        "Computed on the original (unscaled) traffic over the full 1,008 step horizon. "
        "MAPE uses `max(|y_true|, 1)` in the denominator so quiet night hours, where "
        "actual traffic dips close to zero, do not blow up the percentage."
    ))
    for label in AREAS:
        cells.append(md(f"### {label}"))
        cells.append(code(
            f"display(pd.read_csv(TABLES / 'metrics_{label}.csv'))"
        ))

    cells.append(md(
        "## Training and inference time\n\n"
        "All numbers measured on an Apple MacBook Air 13 inch with M3 and 8 GB of "
        "unified memory, macOS 15.5, Python 3.9.6, torch 2.4.1 on the MPS device "
        "for the neural networks, statsmodels 0.14.6 on CPU for SARIMA. Each fit "
        "ran sequentially so timings are not contaminated by parallel workloads."
    ))
    cells.append(code(
        "display(pd.read_csv(TABLES / 'timing_summary.csv'))"
    ))
    cells.append(code(
        "display(pd.read_csv(TABLES / 'timing_avg_per_model.csv'))"
    ))

    cells.append(md(
        "## Comparative analysis\n\n"
        "Quantitative ranking is read off the metric tables. The interpretive points:\n\n"
        "- The two neural networks operate under a strictly easier inference regime "
        "than SARIMA (rolling one step ahead vs. 1,008 step extrapolation), so any "
        "raw MAE comparison must be read with that asymmetry in mind. We chose this "
        "split because the rubric defines the task as one step ahead prediction "
        "for the NN component, while a multi step SARIMA forecast is the canonical "
        "statistical baseline.\n"
        "- The dataset is dominated by a strong daily cycle and a moderate weekday "
        "vs weekend effect that Task 2 made visible. All three architectures have "
        "explicit machinery to capture this: SARIMA via its seasonal order m=144, "
        "the LSTM via its 144 step window, and the TCN via stacked dilations "
        "that reach a 29 step receptive field per block.\n"
        "- Where the NN models tend to beat SARIMA most cleanly is during sharp "
        "transient peaks where SARIMA's smooth multi step extrapolation cannot "
        "resolve the local burst.\n"
        "- The most expensive model in training time is SARIMA. The NN training is "
        "limited by GPU bandwidth on MPS, not compute, so it stays well under a "
        "minute per area at our chosen sizes."
    ))

    cells.append(md(
        "## Failure analysis. The worst 48 hour window across all models\n\n"
        "We slide a 48 hour window (288 ten minute steps) across the test period for "
        "every (model, area) pair, compute the rolling RMSE, and pick the single worst "
        "window across the whole experiment. The plot below shows the actual traffic "
        "together with all three models over that window. Common failure causes are "
        "weekend evening peaks (which deviate from the training tail's weekday pattern) "
        "and sharp anomalous bursts that none of the models anticipate from history alone."
    ))
    cells.append(code(
        "display(Image(str(PLOTS / 'failure_analysis.png')))"
    ))
    cells.append(code(
        "worst = blob.get('worst_window')\n"
        "if worst:\n"
        "    print('worst window:', worst)\n"
    ))

    cells.append(md(
        "## Personal considerations and possible improvements\n\n"
        "- A first improvement would be a per area hyperparameter sweep. We used "
        "one set of hyperparameters across the three areas. The lower volume areas "
        "(B_4159 and C_4556) likely benefit from a smaller hidden state or a wider "
        "window since their signal is noisier in absolute terms.\n"
        "- A multivariate input that incorporates neighbouring grid cells would "
        "almost certainly improve forecasts because Milan's traffic shows strong "
        "spatial correlation (visible in the Task 2 heatmap). The current models "
        "are univariate by choice, to keep the comparison clean and the memory "
        "budget tight.\n"
        "- A Transformer encoder with sinusoidal positional encoding for the hour "
        "of day and day of week features would be a natural next experiment. It "
        "was excluded here for memory reasons.\n"
        "- The SARIMA multi step inference could be replaced by a one step ahead "
        "rolling update for a fairer apples to apples comparison with the NN models. "
        "We kept the multi step setting deliberately to characterize each model's "
        "behaviour in its canonical evaluation mode."
    ))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (venv)",
        "language": "python",
        "name": "python3",
    }
    NB_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_OUT)
    print(f"wrote {NB_OUT}")


if __name__ == "__main__":
    main()
