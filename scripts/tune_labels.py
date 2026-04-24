"""Tune triple-barrier label parameters per instrument on the IS window (2020-2023).

Reads pre-built 15-min bars, runs a grid search over (k_up, k_dn, T, atr_window),
and writes a per-instrument CSV of (combo, class-balance, per-class mean return,
label-return correlation, balance_score).

The 2024+ window is out-of-sample and MUST NOT be passed here.

Usage:
    python scripts/tune_labels.py --instrument ES \
        --bars-glob '/N/project/.../bars/ES/15m/ES_*_15m.parquet' \
        --start 2020-01-01 --end 2023-12-31 \
        --out /N/project/.../label_tuning/ES_tune.csv

Lock the selected combo into configs/label_params.yaml afterwards.
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import date
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.labels.triple_barrier import DEFAULT_COST_PTS, tune_triple_barrier


DEFAULT_K_UP_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
DEFAULT_K_DN_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
DEFAULT_T_GRID = (4, 6, 8, 12, 16, 24)  # 15-min bars: 1h .. 6h horizon
DEFAULT_ATR_WINDOW_GRID = (20, 40, 60)


def _load_bars(bars_glob: str, start: date, end: date) -> pl.DataFrame:
    """Load all 15-min bar parquets matching the glob, filter to [start, end]."""
    paths = sorted(glob.glob(bars_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files match: {bars_glob}")
    print(f"[load] {len(paths)} parquet files matched")

    # Lazy scan + concat so we don't blow memory on ~4yrs × 4 instruments
    frames = [pl.scan_parquet(p) for p in paths]
    lf = pl.concat(frames, how="vertical_relaxed")

    # Filter to IS window. Assume ts is UTC Datetime.
    lf = lf.filter(
        (pl.col("ts").dt.date() >= start) & (pl.col("ts").dt.date() <= end)
    ).sort("ts")

    df = lf.collect()
    print(f"[load] {df.height:,} bars after IS filter [{start} .. {end}]")
    if df.height == 0:
        raise ValueError("Zero bars after IS filter — check bars-glob and date range.")
    return df


def _parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_float_list(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--bars-glob", required=True,
                   help="Glob for per-day 15-min bar parquet files")
    p.add_argument("--start", default="2020-01-01",
                   help="IS window start date (inclusive). Default 2020-01-01.")
    p.add_argument("--end", default="2023-12-31",
                   help="IS window end date (inclusive). Default 2023-12-31. "
                        "MUST be before 2024-01-01 to preserve OOS integrity.")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--k-up-grid", default=None,
                   help="Comma-separated floats; overrides default 1.0,1.25,1.5,1.75,2.0,2.5")
    p.add_argument("--k-dn-grid", default=None)
    p.add_argument("--T-grid", default=None,
                   help="Comma-separated ints; overrides default 4,6,8,12,16,24")
    p.add_argument("--atr-window-grid", default=None,
                   help="Comma-separated ints; overrides default 20,40,60")
    p.add_argument("--cost-pts", type=float, default=None,
                   help="Round-trip cost in price pts (spread+commission+slippage). "
                        "Defaults to DEFAULT_COST_PTS[instrument] (ES:0.50 NQ:1.50 RTY:0.30 YM:3.00).")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end >= date(2024, 1, 1):
        raise SystemExit(
            f"ERROR: --end={end} is in the OOS window (>= 2024-01-01). "
            "Label tuning must use 2020-2023 only."
        )

    k_up_grid = _parse_float_list(args.k_up_grid) if args.k_up_grid else DEFAULT_K_UP_GRID
    k_dn_grid = _parse_float_list(args.k_dn_grid) if args.k_dn_grid else DEFAULT_K_DN_GRID
    T_grid = _parse_int_list(args.T_grid) if args.T_grid else DEFAULT_T_GRID
    atr_window_grid = (
        _parse_int_list(args.atr_window_grid) if args.atr_window_grid else DEFAULT_ATR_WINDOW_GRID
    )
    n_combos = len(k_up_grid) * len(k_dn_grid) * len(T_grid) * len(atr_window_grid)
    cost_pts = args.cost_pts if args.cost_pts is not None else DEFAULT_COST_PTS[args.instrument]
    print(f"[tune] {args.instrument}: grid size {n_combos} combos, cost_pts={cost_pts}")

    bars = _load_bars(args.bars_glob, start, end)

    results = tune_triple_barrier(
        bars,
        k_up_grid=k_up_grid,
        k_dn_grid=k_dn_grid,
        T_grid=T_grid,
        atr_window_grid=atr_window_grid,
        cost_pts=cost_pts,
    )
    results = results.with_columns(pl.lit(args.instrument).alias("instrument"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(out_path)
    print(f"[tune] wrote {out_path}  ({results.height} rows)")

    # Top-10 by balance_score × |label_forward_return_corr| (rough combined objective)
    ranked = (
        results
        .with_columns(
            (pl.col("balance_score") * pl.col("label_forward_return_corr").abs())
            .alias("_combo_score")
        )
        .sort("_combo_score", descending=True)
        .head(10)
    )
    print("\n[tune] top 10 by balance_score × |label_forward_return_corr|:")
    print(ranked.select([
        "instrument", "k_up", "k_dn", "T", "atr_window",
        "frac_pos", "frac_neg", "frac_zero",
        "balance_score", "label_forward_return_corr",
        "mean_ret_pts_pos", "mean_ret_pts_neg",
        "pts_over_cost_pos", "pts_over_cost_neg",
        "n_total",
    ]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
