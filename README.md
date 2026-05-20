# Time Series Forecasting on Milan Mobile Internet Traffic

Formative assignment for ML Techniques I. Comparative analysis and one step ahead forecasting of mobile data traffic on the Telecom Italia Milan grid.

## What is in this repository

| Path | Purpose |
|---|---|
| `src/task1_pipeline.py` | Streaming pipeline that reduces the raw 20.8 GB CDR text files into per square parquet files. |
| `src/task3_forecast.py` | CLI runner that fits SARIMA, LSTM and TCN per area and saves predictions, models, plots, timings. |
| `src/task3_aggregate.py` | Reads saved predictions and produces metric tables, timing tables, the worst window failure plot, and metrics.json. |
| `src/notebook_assembler.py` | Rebuilds `notebooks/task3_modeling.ipynb` from the saved results so the submission has a readable notebook alongside the script. |
| `notebooks/task2_eda.ipynb` | Task 2 exploratory data analysis with all required figures and discussions. |
| `notebooks/task3_modeling.ipynb` | Task 3 modelling notebook regenerated from results. |
| `processed/task1/` | Outputs of Task 1 (`report.md`, per square parquets, spatial grid). |
| `processed/task3/` | Outputs of Task 3 (predictions, models, plots, tables, metrics.json, `report.md`). |
| `reference/milano_grid.geojson` | Geographic grid for the Milan tessellation. |
| `reports/final_report.pdf` | Assembled submission PDF. |
| `datasets/dataverse_files/` | Raw CDR text files (62 daily files). Not committed; download from Harvard Dataverse, see references. |

## Running the code

### 1. Create the environment

The project was developed and tested with Python 3.9 on macOS arm64 (Apple M3).

```bash
python3.9 -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt
```

### 2. Reproduce Task 1 outputs (optional)

The Task 1 outputs are already saved under `processed/task1/`. If you want to regenerate them from the raw CDR files (which must first be downloaded into `datasets/dataverse_files/`):

```bash
python src/task1_pipeline.py
```

Expect roughly 90 to 120 seconds on an Apple M3 with 8 GB unified memory.

### 3. Run Task 3 forecasting

The runner is memory disciplined. One model fits at a time, the previous model is freed before the next starts, predictions are saved to disk immediately so a crash in the middle does not wipe everything.

```bash
# everything (SARIMA + LSTM + TCN, all three areas)
python src/task3_forecast.py --model all

# one model at a time, useful for limited memory machines
python src/task3_forecast.py --model sarima
python src/task3_forecast.py --model lstm
python src/task3_forecast.py --model tcn

# limit to specific areas
python src/task3_forecast.py --model lstm --areas A_5161,B_4159

# if SARIMA with daily seasonality is too slow on your hardware, fall back
python src/task3_forecast.py --model sarima --sarima-fallback
```

### 4. Aggregate metrics, build tables, run failure analysis

```bash
python src/task3_aggregate.py
```

### 5. Regenerate the executed notebook

```bash
python src/notebook_assembler.py
```

## Hardware used for the reported timings

- Apple MacBook Air 13 inch, M3 chip (2024)
- 8 GB unified memory
- macOS Sequoia 15.5
- Python 3.9.6
- PyTorch 2.4.1 with MPS (Metal Performance Shaders) device for the LSTM and TCN
- statsmodels 0.14.6 on CPU for SARIMA

The full Task 3 run completes in well under one hour on this configuration.

## Citations

- Barlacchi G., De Nadai M., Larcher R., Casella A., Chitic C., Torrisi G., Antonelli F., Vespignani A., Pentland A., Lepri B. *A multi source dataset of urban life in the city of Milan and the Province of Trentino.* Sci Data 2, 150055 (2015). https://doi.org/10.1038/sdata.2015.55
- Telecommunications activity dataset for Milan, Harvard Dataverse, doi:10.7910/DVN/EGZHFV
- Milan grid dataset, Harvard Dataverse, doi:10.7910/DVN/QJWLFU

## AI usage statement

Code structure, model templates and report drafting were assisted with an AI coding tool. All design decisions (model choice, hyperparameters, evaluation protocol, file layout, fallback strategy under memory pressure) were made by the author. All implementation outputs were reviewed and validated against the rubric and the actual data.
