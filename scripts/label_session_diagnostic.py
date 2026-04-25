"""Per-hour-of-day diagnostic for triple-barrier labels.

Loads 15-min OHLCV bars for an instrument, runs the labeler with the locked
V1 params, and slices class distribution + return correlation by ET hour.

Supports both calendar ATR (legacy) and time-conditional ATR (TC-ATR) via
--atr-mode. The headline value of this script is comparing the per-hour
class distribution under both modes — TC-ATR should flatten frac_zero
across the 23h CME session.

Usage:
    python scripts/label_session_diagnostic.py \
        --instrument ES \
        --bars-glob '/N/.../label_tuning/bars_ohlcv/ES/15m/*.parquet' \
        --out       /N/.../label_tuning/results/ES_session_diag.csv \
        --atr-mode  time_conditional --lookback-days 30
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.labels.triple_barrier import _balance_score, triple_barrier_labels


# Locked V1 params from LABEL_TUNING_RESULTS.md
V1_PARAMS = {
    'ES':  dict(k_up=1.25, k_dn=1.25, T=8, atr_window=60),
    'NQ':  dict(k_up=1.00, k_dn=1.00, T=6, atr_window=60),
    'RTY': dict(k_up=1.00, k_dn=1.00, T=6, atr_window=60),
    'YM':  dict(k_up=1.25, k_dn=1.25, T=8, atr_window=60),
}

# US equity-index futures session boundaries in ET (US/Eastern).
# CME ES has a daily maintenance halt 17:00-18:00 ET.
SESSION_BOUNDS_ET = [
    ("ASIA",     18, 3),    # 18:00 prev day → 03:00 ET
    ("EU",        3, 9),    # 03:00 → 09:00 ET
    ("US_RTH",    9, 16),   # 09:00 → 16:00 ET (cash session)
    ("US_ETH",   16, 18),   # 16:00 → 18:00 ET (extended-hours wind-down)
]


def _load_bars(bars_glob: str) -> pl.DataFrame:
    paths = sorted(glob.glob(bars_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files match: {bars_glob}")
    print(f"[load] {len(paths)} parquet files")
    df = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical_relaxed").sort("ts").collect()
    print(f"[load] {df.height:,} bars")
    return df


def _classify_session(hour_et: int) -> str:
    for name, lo, hi in SESSION_BOUNDS_ET:
        if lo > hi:  # wraps midnight (ASIA)
            if hour_et >= lo or hour_et < hi:
                return name
        else:
            if lo <= hour_et < hi:
                return name
    return "UNKNOWN"


def per_hour_stats(labeled: pl.DataFrame) -> pl.DataFrame:
    """Slice the labeled frame by hour-of-day (ET) and compute class stats."""
    valid = labeled.filter(
        pl.col("atr").is_finite() & pl.col("realized_ret").is_finite()
    )
    # Convert ts (UTC) to US/Eastern hour and tag with session bucket.
    valid = valid.with_columns(
        pl.col("ts").dt.convert_time_zone("US/Eastern").dt.hour().alias("hour_et")
    )
    valid_pd = valid.with_columns(
        pl.col("hour_et").map_elements(_classify_session, return_dtype=pl.Utf8).alias("session")
    )

    rows = []
    for hour in sorted(valid_pd["hour_et"].unique().to_list()):
        sub = valid_pd.filter(pl.col("hour_et") == hour)
        n = sub.height
        if n == 0:
            continue
        cm = {int(r["label"]): int(r["cnt"]) for r in
              sub.group_by("label").agg(pl.len().alias("cnt")).iter_rows(named=True)}
        n_pos, n_neg, n_zero = cm.get(1, 0), cm.get(-1, 0), cm.get(0, 0)
        fp, fn, fz = n_pos / n, n_neg / n, n_zero / n

        def _mean_pts(lbl: int) -> float:
            s = sub.filter(pl.col("label") == lbl)["realized_ret_pts"]
            return float(s.mean()) if s.len() > 0 else float("nan")

        lbl_arr = sub["label"].cast(pl.Float64).to_numpy()
        ret_arr = sub["realized_ret"].to_numpy()
        if np.std(lbl_arr) > 0 and np.std(ret_arr) > 0:
            corr = float(np.corrcoef(lbl_arr, ret_arr)[0, 1])
        else:
            corr = float("nan")

        rows.append(dict(
            hour_et=hour,
            session=sub["session"][0],
            n=n,
            frac_pos=fp, frac_neg=fn, frac_zero=fz,
            balance_score=_balance_score(fp, fn, fz),
            corr=corr,
            mean_ret_pts_pos=_mean_pts(1),
            mean_ret_pts_neg=_mean_pts(-1),
            mean_ret_pts_zero=_mean_pts(0),
        ))
    return pl.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--bars-glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--atr-mode", default="calendar",
                   choices=["calendar", "time_conditional"],
                   help="ATR computation mode for the labeler")
    p.add_argument("--lookback-days", type=int, default=30,
                   help="Lookback days for time_conditional ATR")
    p.add_argument("--halt-aware", action="store_true", default=True,
                   help="Drop bars whose forward T-window crosses a halt (>30min ts gap)")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end >= date(2024, 1, 1):
        raise SystemExit(f"--end {end} hits OOS window; must be <= 2023-12-31")

    params = V1_PARAMS[args.instrument]
    print(f"[diag] {args.instrument}  params={params}  IS=[{start} .. {end}]  "
          f"atr_mode={args.atr_mode}  lookback_days={args.lookback_days}  "
          f"halt_aware={args.halt_aware}")

    bars = _load_bars(args.bars_glob)
    bars = bars.filter(
        (pl.col("ts").dt.date() >= start) & (pl.col("ts").dt.date() <= end)
    ).sort("ts")
    print(f"[diag] {bars.height:,} bars after IS filter")

    labeled = triple_barrier_labels(
        bars, **params,
        atr_mode=args.atr_mode,
        lookback_days=args.lookback_days,
        halt_aware=args.halt_aware,
    )
    diag = per_hour_stats(labeled)

    # Session-level aggregate (weighted by n)
    by_session = (
        diag.with_columns(
            (pl.col("frac_zero") * pl.col("n")).alias("_fz_w"),
            (pl.col("balance_score") * pl.col("n")).alias("_bs_w"),
        )
        .group_by("session")
        .agg([
            pl.col("n").sum().alias("n"),
            (pl.col("_fz_w").sum() / pl.col("n").sum()).alias("frac_zero_avg"),
            (pl.col("_bs_w").sum() / pl.col("n").sum()).alias("balance_avg"),
            pl.col("corr").mean().alias("corr_unweighted_mean"),
        ])
        .sort("n", descending=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    diag.write_csv(out)
    print(f"\n[diag] per-hour table → {out}\n")
    pl.Config.set_tbl_rows(30).set_tbl_cols(15).set_tbl_width_chars(180)
    print(diag)
    print(f"\n[diag] session-aggregate ({args.instrument}):")
    print(by_session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
