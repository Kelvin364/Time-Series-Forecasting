# Task 3. Time Series Forecasting on the Milan Internet Traffic

Forecast horizon: 2013 12 16 00:00 Europe/Rome to 2013 12 22 23:50. Exactly 1,008 ten minute steps. Training spans 2013 11 01 to 2013 12 15. The last 15 days of the training portion (2013 12 01 to 2013 12 15) are reserved as a validation tail for the neural network models, used only for early stopping. No test data ever enters training.

Targets:

| Area | Square ID | Source |
|---|---|---|
| A_5161 | 5161 | Identified in Task 1 as the highest total traffic square over the full two months |
| B_4159 | 4159 | Specified by the rubric |
| C_4556 | 4556 | Specified by the rubric |

## I. Self contained description of the three models

Three models are fit independently per area (nine fits total) on the cleaned ten minute internet activity series produced by Task 1.

### Model 1. SARIMA (statistical baseline)

The original plan was a statsmodels SARIMAX with daily seasonality, order (1,1,1)(1,1,1,144), `simple_differencing=True`. After the first fit exceeded seven minutes of wall time on the 8 GB M3 with no signs of convergence, the runner was switched to its built in fallback. The fallback combines two pieces:

- **ARIMA(2,1,2)** without seasonal terms, fit on the full 6,480 step training series. This captures local momentum and slow trend.
- **A 144 step seasonal naive** that uses the last full day of training as the next day's prediction and tiles it forward across the test horizon. This captures the dominant daily cycle directly from the training tail without paying the SARIMAX seasonal cost.

The two predictions are blended with equal weights (0.5 each) and clipped at zero. The blend is a classical pattern in baseline forecasting because seasonal naive on its own captures the cycle but misses any drift, while ARIMA on its own captures drift but flattens out for any horizon greater than a few steps.

Inference is multi step: the model is fit once on the training data, then asked to forecast the next 1,008 steps in one go. The model never sees a single test value. This is the hardest evaluation setting and gives us a sober baseline for how well a classical method can do.

Persisted artifacts: `processed/task3/models/sarima_<label>.pkl` (params only, very small).

### Model 2. LSTM

Two layer stacked LSTM, hidden size 64, dropout 0.2, with a Dense(1) head reading the last hidden state. Implemented in PyTorch and trained on the MPS device (Apple Neural / GPU back end on the M3).

- **Input.** A sliding window of 144 scaled values (one day of context).
- **Preprocessing.** MinMaxScaler fit only on the training portion (persisted to `processed/task3/scalers/<label>.pkl`). Inputs are scaled to [0, 1].
- **Optimiser.** Adam with lr=1e 3.
- **Scheduler.** ReduceLROnPlateau with factor 0.5 and patience 3 on validation MSE.
- **Early stopping.** Patience 5 epochs on validation MSE.
- **Capacity.** Max 20 epochs, batch size 64.
- **Inference.** Rolling one step ahead. At each test step the model sees the true past 144 values and predicts the next step. The predicted value is not fed back, the next prediction uses the next true window.

### Model 3. TCN (Temporal Convolutional Network)

Three residual blocks of dilated causal Conv1D, kernel size 3, 32 channels per block, dilation rates (1, 2, 4), dropout 0.2, weight normalisation on each conv. A Dense(1) head reads the last position of the final feature map.

The architecture mirrors Bai et al. 2018 with a smaller channel width for the M3 memory budget. The effective receptive field of one block stack with these dilations is 1 + 2*(3 - 1)*(1+2+4) = 29 steps, but the input window is 144 steps long, so the model sees a full day of context every time.

Same training loop, same scaler, same window length and same rolling one step ahead inference as the LSTM.

## II. Forecast plots (nine plots)

Saved to `processed/task3/plots/forecast_<model>_<area>.png`. Each plot:

- Solid blue: actual ten minute internet activity
- Dashed orange: model prediction
- Weekends (Sat 21, Sun 22) shaded grey
- Title carries MAE, RMSE and MAPE for the (model, area) pair

Files:

```
forecast_sarima_A_5161.png   forecast_lstm_A_5161.png   forecast_tcn_A_5161.png
forecast_sarima_B_4159.png   forecast_lstm_B_4159.png   forecast_tcn_B_4159.png
forecast_sarima_C_4556.png   forecast_lstm_C_4556.png   forecast_tcn_C_4556.png
```

## III. Per area metric tables (Task 3 part III)

Computed on the original (unscaled) traffic over the full 1,008 step test horizon. MAPE uses `max(|y_true|, 1)` in the denominator to avoid percentage blow up where actual traffic dips close to zero (common during quiet night hours). Tables are written to `processed/task3/tables/metrics_<label>.{csv,md}`.

## IV. Training and inference times

`processed/task3/tables/timing_summary.csv` carries one row per (area, model) with `Train (s)`, `Inference (s)`, and `Steps per sec`. `timing_avg_per_model.csv` averages train time per model across the three areas.

All numbers were measured on an Apple MacBook Air 13 inch with M3 chip and 8 GB unified memory, macOS 15.5, Python 3.9.6, torch 2.4.1 on the MPS device for the neural networks, statsmodels 0.14.6 on CPU for the SARIMA baseline. Each fit ran sequentially in its own process; nothing else CPU intensive was running on the machine during measurement.

## V. Personal considerations

The most interesting trade off in this experiment is between the inference regimes of the three models. The two neural networks operate in a rolling one step ahead setting, which is exactly what the rubric defines, while SARIMA runs as a multi step forecast over the full horizon. This means the NN models always know the previous true value at prediction time, while SARIMA's first prediction has to anchor every subsequent prediction. SARIMA's MAPE numbers reflect this disadvantage.

What I would change with more time:
- A per area hyperparameter search. The three areas have very different scales (mean traffic differs by roughly 5x between A_5161 and B_4159) and the optimal hidden size, window length and learning rate are unlikely to be identical across them.
- Multivariate input that includes neighbouring grid cells. Task 2 made it clear that Milan's traffic is strongly spatially correlated. A model with even a small spatial neighbourhood as additional input would almost certainly improve on the univariate baselines.
- Time of day and day of week features. The TCN in particular would benefit from sinusoidal encodings of these because its receptive field is finite.
- An autoregressive SARIMA on the order of a few hours, run in a rolling one step ahead fashion to make the comparison apples to apples. We deliberately picked the multi step setting to characterise SARIMA at its hardest.

## VI. Input representation, preprocessing and normalisation

| Model | Sequence length | Scaling | Differencing | Other |
|---|---|---|---|---|
| SARIMA | full training series | none, traffic on its own scale | first order regular differencing (d=1) inside the ARIMA part; seasonal naive uses raw values | predictions clipped at zero before evaluation |
| LSTM | 144 past steps (one day) | MinMax to [0, 1], fit on training only | none | predictions inverse transformed and clipped at zero |
| TCN | 144 past steps (one day) | MinMax to [0, 1], fit on training only | none | weight normalisation on each Conv1D |

## VII. Comparative analysis

The full quantitative table is in `processed/task3/tables/`. The qualitative picture:

- **SARIMA is fast and cheap** (sub second fit in the fallback variant, sub millisecond inference) but its multi step extrapolation cannot adapt to local bursts. Its errors are largest at the daily evening peak.
- **LSTM and TCN benefit from the rolling regime**. Both are typically within a few percent MAPE of each other and both clearly beat SARIMA on metrics that punish local error (MAE, RMSE), especially on the lower volume areas B_4159 and C_4556 where the SARIMA blend has higher relative noise.
- **TCN trains and runs faster than LSTM** on this machine because convolutions parallelise more cleanly on the MPS device than the sequential LSTM recurrence. For deployments where inference latency matters and accuracy is comparable, the TCN is the natural pick.

The recommended best performing model overall is the one with the lowest mean MAE across the three areas. See `processed/task3/metrics.json` for the consolidated numeric blob and `tables/metrics_<label>.csv` for the per area tables.

## VIII. Failure analysis. The worst 48 hour window

`processed/task3/plots/failure_analysis.png` shows the actual traffic together with all three model predictions over the single worst 48 hour window across all (model, area) pairs. The window was selected by sliding a 288 step rolling RMSE across the test period and picking the highest peak.

Common failure causes visible in the plot:

- **Sharp evening peaks** that occur slightly earlier or later than the training tail's average. SARIMA cannot resolve them at all; the NN models track the rising edge but often overshoot the peak.
- **Weekend traffic patterns** that diverge from the weekday template. Saturday 21 and Sunday 22 December are inside the test window and have a softer daily peak than the training week. All three models tend to over predict the Saturday evening because the most recent training tail was a weekday.
- **Holiday week effects**. The test period sits in the run up to Christmas. Traffic patterns in that week deviate from the typical weekday rhythm seen in November and early December (visible in the EDA anomaly section, items 23 to 26 December dip noticeably).

## Output index

```
src/task3_forecast.py             ← CLI runner that produced everything below
src/task3_aggregate.py            ← builds the tables and the failure plot
src/notebook_assembler.py         ← regenerates the notebook from results
notebooks/task3_modeling.ipynb    ← readable notebook artifact for the submission
processed/task3/
  predictions/<model>_<label>.parquet   ← 1,008 step forecast per (model, area)
  models/<model>_<label>.{pkl|pt}       ← fitted SARIMA params / NN state_dict
  scalers/<label>.pkl                    ← per area MinMaxScaler (training only fit)
  plots/forecast_<model>_<label>.png     ← 9 forecast plots
  plots/failure_analysis.png             ← worst 48 hour window with all three models
  tables/metrics_<label>.{csv|md}        ← per area metric tables
  tables/timing_summary.csv              ← (area, model) train / infer time
  tables/timing_avg_per_model.csv        ← average train time per model
  logs/<model>_<label>_history.csv       ← per epoch train / val loss for NN models
  logs/run.log                           ← full streaming stdout from the runner
  metrics.json                           ← consolidated metrics + timings blob
  report.md                              ← this file
```
