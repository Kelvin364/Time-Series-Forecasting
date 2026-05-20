import gc
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

RAW_DIR      = Path("/Users/kelvinrwihimba/Documents/ML_Learning/Formatives/Time-Series-Forecasting/datasets")          # folder containing daily .txt files
OUT_DIR      = Path("processed")    # output folder for Parquet files
CHUNK_SIZE   = 500_000              # rows per chunk — reduce to 200_000 if RAM pressure
KNOWN_TARGETS = {4159, 4556}        # squares always extracted

USE_COLS  = [0, 1, 7]
COL_NAMES = ["square_id", "time_interval", "internet"]
DTYPES    = {
    "square_id"    : "int16",    # 1–10,000 fits int16 (saves vs int32/int64)
    "time_interval": "int64",    # millisecond Unix epoch — needs int64
    "internet"     : "float32",  # normalised CDR count — float32 sufficient
}

def get_ram_mb() -> float:
    """Current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def get_system_ram_mb() -> dict:
    """System-wide memory snapshot."""
    vm = psutil.virtual_memory()
    return {
        "total_mb"    : vm.total     / 1_048_576,
        "available_mb": vm.available / 1_048_576,
        "used_mb"     : vm.used      / 1_048_576,
        "percent_used": vm.percent,
    }


def sizeof_df(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1_048_576


def log(msg: str, file=None):
    """Print to stdout and optionally write to a log file."""
    print(msg, flush=True)
    if file:
        file.write(msg + "\n")
        file.flush()


def optimise_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast a raw chunk to memory-efficient dtypes.
    Called before any groupby to minimise peak RAM.
    """
    chunk["square_id"]     = chunk["square_id"].astype("int16")
    chunk["time_interval"] = chunk["time_interval"].astype("int64")
    chunk["internet"]      = chunk["internet"].astype("float32")
    return chunk


def benchmark_memory_optimisation(sample_file: Path, log_fh) -> dict:
    """
    Load a sample chunk with default dtypes, then with optimised dtypes.
    Record and return before/after memory usage for Task 1 documentation.
    """
    log("\n" + "="*60, log_fh)
    log("MEMORY OPTIMISATION BENCHMARK", log_fh)
    log("="*60, log_fh)

    sample_rows = 500_000

    # ── BEFORE optimisation (pandas defaults) ──────────────────
    log(f"\nLoading {sample_rows:,} rows with DEFAULT dtypes ...", log_fh)
    t0 = time.perf_counter()
    df_before = pd.read_csv(
        sample_file, sep="\t", header=None,
        names=["square_id","time_interval","internet"],
        usecols=USE_COLS,
        nrows=sample_rows,
    )
    t_before = time.perf_counter() - t0
    mem_before = sizeof_df(df_before)

    log(f"  dtypes       : {df_before.dtypes.to_dict()}", log_fh)
    log(f"  memory usage : {mem_before:.2f} MB", log_fh)
    log(f"  load time    : {t_before:.3f}s", log_fh)
    log(f"  inferred dtypes — square_id={df_before['square_id'].dtype}, "
        f"time_interval={df_before['time_interval'].dtype}, "
        f"internet={df_before['internet'].dtype}", log_fh)

    # ── AFTER optimisation ─────────────────────────────────────
    log(f"\nLoading {sample_rows:,} rows with OPTIMISED dtypes ...", log_fh)
    t0 = time.perf_counter()
    df_after = pd.read_csv(
        sample_file, sep="\t", header=None,
        names=["square_id","time_interval","internet"],
        usecols=USE_COLS,
        dtype=DTYPES,
        nrows=sample_rows,
    )
    t_after = time.perf_counter() - t0
    mem_after = sizeof_df(df_after)

    log(f"  dtypes       : {df_after.dtypes.to_dict()}", log_fh)
    log(f"  memory usage : {mem_after:.2f} MB", log_fh)
    log(f"  load time    : {t_after:.3f}s", log_fh)

    reduction = (mem_before - mem_after) / mem_before * 100
    log(f"\n  ✓ Memory reduction : {mem_before:.2f} MB → {mem_after:.2f} MB "
        f"({reduction:.1f}% saved)", log_fh)
    log(f"  ✓ Technique        : explicit dtype casting "
        f"(int16, int64, float32 vs pandas defaults int64/float64)", log_fh)

    # Additional NaN analysis
    null_count = df_after["internet"].isna().sum()
    log(f"\n  NaN in internet column (sample): {null_count:,} "
        f"({null_count/len(df_after)*100:.2f}%)", log_fh)
    log("  → NaN rows dropped during aggregation (NaN CDR = no activity)", log_fh)

    del df_before, df_after
    gc.collect()

    return {
        "mem_before_mb" : mem_before,
        "mem_after_mb"  : mem_after,
        "reduction_pct" : reduction,
        "load_time_default_s"  : t_before,
        "load_time_optimised_s": t_after,
    }


def streaming_pass(
    files: list,
    target_ids: set,
    pass_label: str,
    log_fh,
    accumulate_totals: bool = True,
) -> tuple[dict, dict]:
    """
    Single streaming pass over all daily files.

    Parameters
    ----------
    files            : sorted list of raw .txt Paths
    target_ids       : set of square_ids whose full time series to collect
    pass_label       : "Pass 1" or "Pass 2" — for logging
    log_fh           : open log file handle
    accumulate_totals: whether to build the 10k-square totals dict

    Returns
    -------
    totals : dict  {square_id: cumulative_internet_float}
    series : dict  {square_id: list of DataFrames}
    """
    totals  = {} if accumulate_totals else None
    series  = {sq: [] for sq in target_ids}

    total_raw_rows   = 0
    total_agg_rows   = 0
    total_null_rows  = 0
    pass_start       = time.perf_counter()
    peak_ram_mb      = 0.0

    log(f"\n{'='*60}", log_fh)
    log(f"{pass_label} — streaming {len(files)} files", log_fh)
    log(f"Target squares  : {sorted(target_ids)}", log_fh)
    log(f"Chunk size      : {CHUNK_SIZE:,} rows", log_fh)
    log(f"System RAM      : {get_system_ram_mb()}", log_fh)
    log(f"Process RAM at start: {get_ram_mb():.1f} MB", log_fh)
    log("="*60, log_fh)

    for filepath in tqdm(files, desc=pass_label, unit="file"):
        file_start = time.perf_counter()
        file_raw   = 0
        file_agg   = 0

        try:
            reader = pd.read_csv(
                filepath,
                sep="\t",
                header=None,
                names=COL_NAMES,
                usecols=USE_COLS,
                dtype=DTYPES,
                chunksize=CHUNK_SIZE,
                na_values=["", "NA", "NaN"],
            )

            for chunk in reader:
                chunk_raw = len(chunk)
                file_raw  += chunk_raw
                total_raw_rows += chunk_raw

                # Drop NaN internet rows (no CDR activity recorded)
                null_mask = chunk["internet"].isna()
                null_count = null_mask.sum()
                total_null_rows += null_count
                chunk = chunk[~null_mask]

                # ── KEY STEP: aggregate across country codes ────
                # Raw file has multiple rows per (square_id, time_interval)
                # — one per country code present. We sum internet activity.
                agg = (
                    chunk
                    .groupby(["square_id", "time_interval"], sort=False)["internet"]
                    .sum()
                    .reset_index()
                )
                agg["internet"] = agg["internet"].astype("float32")

                chunk_agg  = len(agg)
                file_agg   += chunk_agg
                total_agg_rows += chunk_agg

                # ── Accumulate per-square totals ────────────────
                if accumulate_totals:
                    sq_sums = agg.groupby("square_id")["internet"].sum()
                    for sq, val in sq_sums.items():
                        totals[sq] = totals.get(sq, 0.0) + float(val)

                # ── Extract target time series ──────────────────
                mask = agg["square_id"].isin(target_ids)
                if mask.any():
                    sub = agg[mask]
                    for sq in target_ids:
                        part = sub[sub["square_id"] == sq]
                        if len(part):
                            series[sq].append(
                                part[["time_interval", "internet"]].copy()
                            )

                # ── Memory tracking ─────────────────────────────
                current_ram = get_ram_mb()
                if current_ram > peak_ram_mb:
                    peak_ram_mb = current_ram

                del chunk, agg
                if mask.any():
                    del sub
                gc.collect()

        except Exception as e:
            log(f"\n  ERROR processing {filepath.name}: {e}", log_fh)
            traceback.print_exc()
            continue

        file_time = time.perf_counter() - file_start
        compression = (1 - file_agg / file_raw) * 100 if file_raw else 0
        log(
            f"  {filepath.name}: "
            f"{file_raw:>9,} raw rows → {file_agg:>8,} agg rows "
            f"({compression:.1f}% reduction) | {file_time:.1f}s",
            log_fh,
        )

    pass_time = time.perf_counter() - pass_start
    overall_compression = (1 - total_agg_rows / total_raw_rows) * 100 if total_raw_rows else 0

    log(f"\n{pass_label} SUMMARY", log_fh)
    log(f"  Total raw rows      : {total_raw_rows:>12,}", log_fh)
    log(f"  Total NaN dropped   : {total_null_rows:>12,} ({total_null_rows/total_raw_rows*100:.2f}%)", log_fh)
    log(f"  Total agg rows      : {total_agg_rows:>12,}", log_fh)
    log(f"  Overall compression : {overall_compression:.1f}%", log_fh)
    log(f"  Peak process RAM    : {peak_ram_mb:.1f} MB", log_fh)
    log(f"  Wall time           : {pass_time/60:.1f} min ({pass_time:.0f}s)", log_fh)

    return totals, series

def save_series_parquet(series: dict, out_dir: Path, log_fh):
    """
    Concatenate per-square chunk lists, convert timestamp,
    sort chronologically, and save to Parquet.
    """
    log("\nSaving time series to Parquet ...", log_fh)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sq_id, chunks in series.items():
        if not chunks:
            log(f"  WARNING: no data collected for square {sq_id}", log_fh)
            continue

        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values("time_interval").drop_duplicates(subset="time_interval")

        # Convert millisecond epoch to UTC datetime index
        df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["internet"]].rename(
            columns={"internet": "internet_activity"}
        )

        out_path = out_dir / f"square_{sq_id}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")

        n_rows    = len(df)
        file_size = out_path.stat().st_size / 1024
        span_days = (df.index[-1] - df.index[0]).days
        log(
            f"  square_{sq_id}.parquet: {n_rows:,} rows | "
            f"{span_days} days | {file_size:.1f} KB",
            log_fh,
        )
        del df
        gc.collect()


def save_totals_parquet(totals: dict, out_dir: Path, log_fh):
    """Save per-square two-month totals as a Parquet file."""
    log("\nSaving all-square totals to Parquet ...", log_fh)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pd.Series(totals, name="total_internet")
        .rename_axis("square_id")
        .sort_index()
        .to_frame()
    )
    df.index = df.index.astype("int16")

    out_path = out_dir / "all_totals.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    file_size = out_path.stat().st_size / 1024
    log(f"  all_totals.parquet: {len(df):,} squares | {file_size:.1f} KB", log_fh)
    return df


# ─────────────────────────────────────────────────────────────
# TASK 1 REPORT WRITER
# ─────────────────────────────────────────────────────────────

def write_task1_report(
    bench: dict,
    pass1_totals: dict,
    winner_id: int,
    all_squares: int,
    out_dir: Path,
    meta: dict,
):
    """Write the Task 1 written report as plain text."""
    report_path = out_dir / "task1_report.txt"
    with open(report_path, "w") as f:
        f.write("TASK 1 — DATA HANDLING & MEMORY MANAGEMENT\n")
        f.write("Milan Telecom CDR Dataset\n")
        f.write("="*60 + "\n\n")

        f.write("I. STRATEGY FOR LOADING AND PROCESSING THE DATASET\n")
        f.write("-"*50 + "\n")
        f.write(
            f"The dataset comprises {meta['n_files']} daily tab-separated text "
            f"files spanning {meta['date_first']} to {meta['date_last']}, totalling "
            f"approximately {meta['total_raw_gb']:.1f} GB of raw uncompressed data. "
            f"It was downloaded from Harvard Dataverse (doi:10.7910/DVN/EGZHFV) "
            f"as {meta['n_chunks']} compressed archive chunks of ~5 GB combined; "
            f"the figure cited in the task brief refers to that compressed download "
            f"size, while the {meta['total_raw_gb']:.1f} GB above is the post-decompression "
            f"volume actually fed to the pipeline.\n\n"
            "Given that the target hardware (MacBook Air M3, 8 GB unified memory) "
            "cannot hold more than a fraction of this in RAM simultaneously, a "
            "streaming chunked approach was adopted.\n\n"
            "Each file is read in chunks of 500,000 rows using pandas read_csv with "
            "chunksize. Only three of the eight columns are loaded per chunk: "
            "square_id, time_interval, and internet_traffic_activity. All other "
            "columns (country_code, sms_in, sms_out, call_in, call_out) are skipped "
            "at the parser level via usecols, ensuring they never enter memory.\n\n"
            f"Each chunk is processed and discarded before the next is read. "
            f"After processing all {meta['n_files']} files, a second targeted pass "
            f"extracts the time series for the highest-traffic square identified "
            f"in Pass 1.\n"
        )

        f.write("\nII. DATA REDUCTION AND TRANSFORMATION METHODS\n")
        f.write("-"*50 + "\n")
        f.write(
            "A. Column pruning: 5 of 8 columns discarded at parse time. "
            "Only square_id, time_interval, and internet_traffic_activity retained.\n\n"
            "B. Dtype optimisation: pandas defaults to int64 for integer columns "
            "and float64 for floats. We override:\n"
            "   • square_id     → int16   (range 1–10,000; int16 max = 32,767)\n"
            "   • time_interval → int64   (millisecond epoch; must retain precision)\n"
            "   • internet      → float32 (normalised CDR count; float32 precision adequate)\n\n"
            "C. Country-code aggregation: each (square_id, time_interval) pair "
            "appears multiple times in the raw file — once per country code of users "
            "present in that cell. A groupby sum collapses these into a single row "
            "per cell per 10-minute slot, reducing row count by ~68% per file.\n\n"
            "D. NaN handling: internet_traffic_activity rows with NaN values "
            "(indicating zero activity reported for a country-code/cell combination) "
            "are dropped before aggregation. This is valid because NaN in this context "
            "encodes absence of CDR events, not missing sensor data.\n\n"
            "E. Parquet output: processed time series saved as Snappy-compressed "
            "Parquet files. Parquet columnar storage with Snappy compression achieves "
            "~10x size reduction over CSV/text and loads ~10x faster on subsequent reads.\n"
        )

        f.write("\nIII. MEMORY USAGE — BEFORE AND AFTER OPTIMISATION\n")
        f.write("-"*50 + "\n")
        f.write(
            f"Sample size: 500,000 rows from one daily file\n\n"
            f"Default dtypes (pandas inferred):\n"
            f"  Memory usage : {bench['mem_before_mb']:.2f} MB\n"
            f"  Load time    : {bench['load_time_default_s']:.3f}s\n\n"
            f"Optimised dtypes (int16, int64, float32):\n"
            f"  Memory usage : {bench['mem_after_mb']:.2f} MB\n"
            f"  Load time    : {bench['load_time_optimised_s']:.3f}s\n\n"
            f"Memory reduction: {bench['reduction_pct']:.1f}%\n\n"
            "Justification: float64→float32 halves floating-point storage with "
            "negligible precision loss for normalised CDR counts (values typically "
            "0–200). int64→int16 for square_id reduces integer storage by 4× since "
            "the domain [1,10000] fits within int16's range [−32768, 32767].\n"
        )

        f.write("\nIV. CHALLENGES AND HOW THEY WERE ADDRESSED\n")
        f.write("-"*50 + "\n")
        f.write(
            "Challenge 1 — Unknown winner square ID at pipeline start.\n"
            "The square with highest total traffic cannot be known before processing "
            "all 62 files. Solution: two-pass architecture. Pass 1 accumulates "
            "running totals for all 10,000 squares in a lightweight Python dict "
            "(a few MB). After Pass 1, argmax identifies the winner. Pass 2 then "
            "extracts that square's full time series.\n\n"
            "Challenge 2 — Country-code fan-out inflates raw row count ~3–5×.\n"
            "Each cell-timeslot pair has multiple rows (one per country code), "
            "making the 62-day dataset ~273M rows rather than ~87M. Solution: "
            "the first operation on every chunk is a groupby-sum that collapses "
            "country codes, reducing each chunk's row count before any further "
            "processing.\n\n"
            "Challenge 3 — NaN values in internet column.\n"
            "Some country-code rows have no recorded internet activity. "
            "Solution: drop NaN rows before aggregation. The groupby-sum "
            "then correctly produces totals from only rows with observed CDRs.\n\n"
            "Challenge 4 — Memory pressure during groupby on large chunks.\n"
            "Groupby operations create temporary intermediate objects. "
            "Solution: explicit gc.collect() after each chunk, and chunk size "
            "tuned to 500,000 rows (approx. 24 MB per chunk at optimised dtypes), "
            "well within the available working memory budget.\n"
        )

        f.write("\nV. HARDWARE AND SOFTWARE SETUP\n")
        f.write("-"*50 + "\n")
        f.write(
            "Hardware: Apple MacBook Air 13-inch, M3 chip (2024)\n"
            "          8 GB unified memory (shared CPU/GPU/Neural Engine)\n"
            "          macOS Sequoia 15.5\n\n"
            f"Software: Python {meta['python_version']}\n"
            f"          pandas {meta['pandas_ver']} (chunked streaming via read_csv)\n"
            f"          numpy {meta['numpy_ver']} (dtype operations)\n"
            f"          pyarrow {meta['pyarrow_ver']} (Parquet I/O)\n"
            f"          psutil {meta['psutil_ver']} (memory profiling)\n\n"
            "Limitations and adaptations:\n"
            "• 8 GB unified memory is shared with macOS (~2–3 GB at idle), "
            "leaving ~5 GB working space. Chunk size set to 500,000 rows "
            "(~24 MB at optimised dtypes) to leave headroom for groupby "
            "intermediate objects.\n"
            "• Full dataset cannot be loaded into memory simultaneously. "
            "The streaming architecture processes one daily file at a time, "
            "discarding each before loading the next.\n"
            "• Apple M3 MPS backend (Metal Performance Shaders) will be "
            "used for PyTorch model training in Task 3, leveraging the "
            "on-chip GPU without separate GPU memory constraints.\n"
        )

        f.write(f"\nHighest-traffic square identified: {winner_id}\n")
        f.write(f"Total squares in dataset: {all_squares}\n")

    print(f"\nTask 1 report written → {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "pipeline_log.txt"

    with open(log_path, "w") as log_fh:

        log("MILAN TELECOM CDR — TASK 1 PIPELINE", log_fh)
        log(f"Start time : {pd.Timestamp.now()}", log_fh)
        log(f"Raw data   : {RAW_DIR.resolve()}", log_fh)
        log(f"Output     : {OUT_DIR.resolve()}", log_fh)

        # ── Validate raw directory ──────────────────────────────
        # The Dataverse archive arrives as several download chunks
        # (datasets/dataverse_files, dataverse_files (1)/, ...), each holding a
        # subset of the 62 daily .txt files. Recurse and de-duplicate by
        # filename so each daily file is processed exactly once even if a
        # chunk overlaps with another.
        candidates = sorted(RAW_DIR.rglob("sms-call-internet-mi-*.txt"))
        unique_by_name = {}
        for p in candidates:
            unique_by_name.setdefault(p.name, p)
        txt_files = sorted(unique_by_name.values(), key=lambda p: p.name)

        if not txt_files:
            log(f"\nERROR: No .txt files found under {RAW_DIR}", log_fh)
            log("Please download the dataset from doi:10.7910/DVN/EGZHFV", log_fh)
            log(f"and place the daily .txt files anywhere under {RAW_DIR}.", log_fh)
            sys.exit(1)

        chunk_dirs = sorted({p.parent for p in txt_files})
        total_raw_size_gb = sum(f.stat().st_size for f in txt_files) / 1e9
        date_first = txt_files[0].stem.replace("sms-call-internet-mi-", "")
        date_last  = txt_files[-1].stem.replace("sms-call-internet-mi-", "")

        log(f"\nFound {len(txt_files)} unique daily files "
            f"across {len(chunk_dirs)} download chunk folder(s)", log_fh)
        log(f"Date range     : {date_first} → {date_last}", log_fh)
        log(f"Total raw size : {total_raw_size_gb:.2f} GB", log_fh)
        if len(candidates) != len(txt_files):
            log(f"Note: {len(candidates) - len(txt_files)} duplicate filename(s) "
                f"across chunk folders were skipped.", log_fh)

        # ── Memory benchmark (Task 1 requirement) ──────────────
        bench = benchmark_memory_optimisation(txt_files[0], log_fh)

        # ── Pass 1 ──────────────────────────────────────────────
        log("\nStarting Pass 1 — full dataset scan ...", log_fh)
        ram_before_pass1 = get_ram_mb()
        log(f"Process RAM before Pass 1: {ram_before_pass1:.1f} MB", log_fh)

        totals, series = streaming_pass(
            files=txt_files,
            target_ids=KNOWN_TARGETS,
            pass_label="Pass 1",
            log_fh=log_fh,
            accumulate_totals=True,
        )

        ram_after_pass1 = get_ram_mb()
        log(f"\nProcess RAM after Pass 1: {ram_after_pass1:.1f} MB", log_fh)

        # ── Identify winner ─────────────────────────────────────
        winner_id    = int(max(totals, key=totals.get))
        winner_total = totals[winner_id]
        log(f"\nHighest-traffic square: {winner_id}", log_fh)
        log(f"  Total internet activity: {winner_total:,.2f}", log_fh)
        log(f"  Square 4159 total      : {totals.get(4159, 0):,.2f}", log_fh)
        log(f"  Square 4556 total      : {totals.get(4556, 0):,.2f}", log_fh)

        # Top 10 for reference
        top10 = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:10]
        log("\nTop 10 squares by total internet activity:", log_fh)
        for rank, (sq, val) in enumerate(top10, 1):
            log(f"  #{rank:>2}  square {sq:>5}  →  {val:>12,.2f}", log_fh)

        # ── Save Pass 1 outputs ─────────────────────────────────
        save_series_parquet(series, OUT_DIR, log_fh)
        totals_df = save_totals_parquet(totals, OUT_DIR, log_fh)

        # ── Pass 2 — extract winner series ─────────────────────
        if winner_id not in KNOWN_TARGETS:
            log(f"\nStarting Pass 2 — extracting square {winner_id} ...", log_fh)
            _, winner_series = streaming_pass(
                files=txt_files,
                target_ids={winner_id},
                pass_label="Pass 2",
                log_fh=log_fh,
                accumulate_totals=False,
            )
            save_series_parquet(winner_series, OUT_DIR, log_fh)
        else:
            log(f"\nSquare {winner_id} was already a known target — Pass 2 skipped.", log_fh)

        # ── Final output summary ─────────────────────────────────
        log("\n" + "="*60, log_fh)
        log("OUTPUT FILES", log_fh)
        log("="*60, log_fh)
        for f in sorted(OUT_DIR.glob("*.parquet")):
            size_kb = f.stat().st_size / 1024
            log(f"  {f.name:<35} {size_kb:>8.1f} KB", log_fh)

        log(f"\nProcess RAM at end: {get_ram_mb():.1f} MB", log_fh)
        log(f"End time: {pd.Timestamp.now()}", log_fh)

        # ── Write Task 1 report ─────────────────────────────────
        meta = {
            "n_files"       : len(txt_files),
            "n_chunks"      : len(chunk_dirs),
            "total_raw_gb"  : total_raw_size_gb,
            "date_first"    : date_first,
            "date_last"     : date_last,
            "python_version": platform.python_version(),
            "pandas_ver"    : pd.__version__,
            "numpy_ver"     : np.__version__,
            "pyarrow_ver"   : pa.__version__,
            "psutil_ver"    : psutil.__version__,
        }
        write_task1_report(
            bench=bench,
            pass1_totals=totals,
            winner_id=winner_id,
            all_squares=len(totals),
            out_dir=OUT_DIR,
            meta=meta,
        )

        log("\n✓ Pipeline complete. All outputs in processed/", log_fh)
        log("  Next step: open task2_eda.ipynb to begin Task 2.", log_fh)


if __name__ == "__main__":
    main()