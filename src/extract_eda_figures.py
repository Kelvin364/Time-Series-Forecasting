"""
Pull figures out of notebooks/task2_eda.ipynb and write them as PNGs into
reports/figures/. Used by the PDF builder.

We look at the outputs of each code cell. The first cell that produces a PNG
maps to the section that cell belongs to. The mapping is heuristic but stable
for this notebook.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB = ROOT / "notebooks" / "task2_eda.ipynb"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Map by section number to the desired filename. The notebook layout (verified
# earlier) places one figure producing cell after each section header in this
# order.
NAMES = [
    "eda_pdf.png",            # I  density of two month total traffic
    "eda_timeseries.png",     # II two week time series of three areas
    "eda_stationarity.png",   # III rolling stats + ADF
    "eda_decomposition.png",  # IV trend, seasonal, residual
    "eda_acf_pacf.png",       # V ACF, PACF
    "eda_heatmap.png",        # VI spatial heatmap
    "eda_anomalies.png",      # VII anomalies
    "eda_weekday_vs_weekend.png",  # supplementary
]


def main():
    nb = json.loads(NB.read_text())
    figs_written = 0
    name_idx = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            png = data.get("image/png")
            if png is None:
                continue
            if name_idx >= len(NAMES):
                break
            target = OUT / NAMES[name_idx]
            target.write_bytes(base64.b64decode(png))
            print("wrote", target)
            figs_written += 1
            name_idx += 1
            break  # one figure per cell is enough
        if name_idx >= len(NAMES):
            break
    print(f"total figures: {figs_written}")


if __name__ == "__main__":
    main()
