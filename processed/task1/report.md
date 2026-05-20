# Task 1. Data Handling and Memory Management

Milan Telecom CDR dataset, two months of mobile internet activity over a 100 by 100 grid.

## I. Strategy for loading and processing the dataset

The dataset comprises 62 daily tab separated text files spanning 2013 11 01 to 2014 01 01, totalling roughly 20.8 GB of raw uncompressed data. It was downloaded from Harvard Dataverse (doi:10.7910/DVN/EGZHFV) as seven compressed archive chunks of about 5 GB combined. The 5 GB figure in the task brief refers to the compressed download size, while the 20.8 GB above is the post decompression volume actually fed to the pipeline.

The target hardware is a MacBook Air M3 with 8 GB of unified memory, so loading the whole dataset at once is not an option. I went with a streaming chunked approach instead.

Each file is read in chunks of 500,000 rows using pandas `read_csv` with `chunksize`. Only three of the eight columns are loaded per chunk: `square_id`, `time_interval`, and `internet_traffic_activity`. Every other column (country code, sms in, sms out, call in, call out) is skipped at the parser level via `usecols` so they never enter memory.

Each chunk is processed and discarded before the next one is read. After all 62 files have been streamed through pass one, a second targeted pass extracts the time series for the highest traffic square identified in pass one.

## II. Data reduction and transformation methods

**A. Column pruning.** Five of the eight columns are dropped at parse time. Only `square_id`, `time_interval`, and `internet_traffic_activity` are retained.

**B. Dtype optimisation.** Pandas defaults to `int64` for integer columns and `float64` for floats. I override these as follows:

- `square_id` &rarr; `int16` (range 1 to 10,000; int16 max is 32,767)
- `time_interval` &rarr; `int64` (millisecond epoch, full precision required)
- `internet` &rarr; `float32` (normalised CDR count, float32 precision is more than adequate here)

**C. Country code aggregation.** Each (square id, time interval) pair shows up multiple times in the raw file, once per country code of the users present in that cell. A `groupby` sum collapses these into a single row per cell per ten minute slot, reducing row count by roughly 68% per file.

**D. NaN handling.** Rows where `internet_traffic_activity` is NaN (which encodes zero activity reported for that country code in that cell) are dropped before aggregation. This is valid because NaN here means absence of CDR events, not missing sensor data.

**E. Parquet output.** Processed time series are saved as Snappy compressed Parquet files. Parquet columnar storage with Snappy gives roughly 10x size reduction over CSV or raw text, and loads about 10x faster on subsequent reads.

## III. Memory usage before and after optimisation

Sample size used for the benchmark: 500,000 rows from one daily file.

| | Default dtypes | Optimised dtypes |
|---|---|---|
| Memory usage | 11.44 MB | 6.68 MB |
| Load time | 0.109 s | 0.106 s |

**Memory reduction: 41.7%.** Justification: `float64` to `float32` halves the floating point footprint with negligible precision loss for normalised CDR counts (values typically 0 to 200). `int64` to `int16` for `square_id` cuts integer storage by 4x because the domain [1, 10000] fits comfortably inside int16's range of [&minus;32768, 32767].

Extrapolated across the full dataset, this is the difference between roughly 5 GB and 3 GB of in memory working data over the streaming run.

## IV. Challenges encountered and how they were addressed

**Challenge 1. Unknown winner square id at pipeline start.**
The square with highest total traffic cannot be known before processing all 62 files. The solution is a two pass architecture. Pass one accumulates running totals for all 10,000 squares in a lightweight Python dict (a few MB). After pass one, an argmax identifies the winner. Pass two then extracts that square's full time series, plus the two squares specified in Task 2 (4159 and 4556).

**Challenge 2. Country code fan out inflates raw row count three to five times.**
Each cell timeslot pair has multiple rows, one per country code, so the 62 day dataset is around 273 million rows rather than the expected 87 million. The first operation on every chunk is therefore a groupby sum that collapses country codes, shrinking each chunk's row count before any further processing.

**Challenge 3. NaN values in the internet column.**
Roughly 48% of rows in a sample chunk have NaN in `internet_traffic_activity`. These are dropped before aggregation. The groupby sum then correctly produces totals from only rows with observed CDRs.

**Challenge 4. Memory pressure during groupby on large chunks.**
Groupby operations create temporary intermediate objects that briefly double or triple memory usage. I added an explicit `gc.collect()` after each chunk and tuned chunk size to 500,000 rows (about 24 MB per chunk at optimised dtypes), which leaves comfortable headroom for those intermediates inside the 5 GB working budget.

## V. Hardware and software setup

**Hardware**
- Apple MacBook Air 13 inch, M3 chip (2024)
- 8 GB unified memory, shared between CPU, GPU and Neural Engine
- macOS Sequoia 15.5

**Software**
- Python 3.9.6
- pandas 2.3.3 (chunked streaming via `read_csv`)
- numpy 2.0.2 (dtype operations)
- pyarrow 21.0.0 (Parquet I/O)
- psutil 7.2.2 (memory profiling)

**Limitations and how they shaped the design**
- 8 GB unified memory is shared with macOS itself (about 2 to 3 GB at idle), leaving roughly 5 GB of working space. Chunk size is set to 500,000 rows (about 24 MB at optimised dtypes) to keep headroom for groupby intermediates.
- The full dataset cannot be loaded into memory simultaneously, so the streaming architecture processes one daily file at a time and discards each one before loading the next.
- For Task 3, the same M3 is used with the MPS backend (Metal Performance Shaders) for PyTorch training, which leverages the on chip GPU without separate GPU memory constraints.

## Result

- Highest traffic square identified: **5161**
- Total squares in dataset: **10,000**
- Total processed rows after aggregation: roughly **89 million** (62 days x 144 ten minute slots x 10,000 cells, minus dropped NaNs)
- Output files (saved to `processed/task1/`):
  - `all_totals.parquet` two month total per square (10,000 rows)
  - `square_5161.parquet`, `square_4159.parquet`, `square_4556.parquet` per square time series (8,928 rows each)
  - `spatial_grid.npy` 100 by 100 grid of totals for the spatial heatmap in Task 2
  - `pipeline_log.txt` full streaming log of the pipeline run
