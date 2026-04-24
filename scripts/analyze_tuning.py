"""Per-year stability analysis for triple-barrier tuning results.

Reads an instrument's tuning CSV (output of tune_labels.py), picks the top-N
combos ranked by `balance_score × |label_forward_return_corr|`, re-runs the
labeler on each IS year (2020, 2021, 2022, 2023) separately, and writes a
per-(combo, year) stability CSV.

Use this to filter out combos that look balanced overall but are actually
getting lucky in one regime while being ~useless in another (e.g., COVID 2020
dominating, or 2022 bear giving unidirectional +1 labels).

Usage:
    python scripts/analyze_tuning.py \
        --instrument ES \
        --bars-glob '/N/project/.../label_tuning/bars_ohlcv/ES/15m/*.parquet' \
        --tune-csv  /N/project/.../results/ES_tune_2020_2023.csv \
        --top 10 \
        --out /N/project/.../results/ES_stability_2020_2023.csv
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

from src.labels.triple_barrier import (
    DEFAULT_COST_PTS,
    _balance_score,
    triple_barrier_labels,
)


IS_YEARS = (2020, 2021, 2022, 2023)


def _load_bars(bars_glob: str) -> pl.DataFrame:
    paths = sorted(glob.glob(bars_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files match: {bars_glob}")
    print(f"[load] {len(paths)} parquet files matched")
    frames = [pl.scan_parquet(p) for p in paths]
    df = pl.concat(frames, how="vertical_relaxed").sort("ts").collect()
    print(f"[load] {df.height:,} bars total")
    return df


def _pick_top_combos(tune_csv: Path, n: int) -> pl.DataFrame:
    """Rank by balance_score × |corr|, return top-n as (k_up, k_dn, T, atr_window)."""
    t = pl.read_csv(tune_csv)
    t = t.with_columns(
        (pl.col("balance_score") * pl.col("label_forward_return_corr").abs())
        .alias("_combo_score")
    )
    return (
        t.sort("_combo_score", descending=True)
        .head(n)
        .select(["k_up", "k_dn", "T", "atr_window", "balance_score",
                 "label_forward_return_corr", "_combo_score"])
    )


def _year_stats(
    bars_year: pl.DataFrame, k_up: float, k_dn: float, T: int, atr_window: int,
    cost_pts: float,
) -> dict:
    labeled = triple_barrier_labels(bars_year, k_up=k_up, k_dn=k_dn, T=T, atr_window=atr_window)
    valid = labeled.filter(
        pl.col("atr").is_not_null() & pl.col("realized_ret").is_not_null()
    )
    n = valid.height
    if n == 0:
        return {"n": 0}
    counts = valid.group_by("label").agg(pl.len().alias("cnt"))
    cm = {int(r["label"]): int(r["cnt"]) for r in counts.iter_rows(named=True)}
    n_pos, n_neg, n_zero = cm.get(1, 0), cm.get(-1, 0), cm.get(0, 0)
    fp, fn, fz = n_pos / n, n_neg / n, n_zero / n

    def _mean_pts(lbl: int) -> float:
        sub = valid.filter(pl.col("label") == lbl)["realized_ret_pts"]
        return float(sub.mean()) if sub.len() > 0 else float("nan")

    lbl_arr = valid["label"].cast(pl.Float64).to_numpy()
    ret_arr = valid["realized_ret"].to_numpy()
    if np.std(lbl_arr) > 0 and np.std(ret_arr) > 0:
        corr = float(np.corrcoef(lbl_arr, ret_arr)[0, 1])
    else:
        corr = float("nan")

    pts_p, pts_n = _mean_pts(1), _mean_pts(-1)
    return {
        "n": n,
        "frac_pos": fp, "frac_neg": fn, "frac_zero": fz,
        "balance_score": _balance_score(fp, fn, fz),
        "corr": corr,
        "mean_ret_pts_pos": pts_p, "mean_ret_pts_neg": pts_n,
        "pts_over_cost_pos": abs(pts_p) / cost_pts if cost_pts > 0 and not np.isnan(pts_p) else float("nan"),
        "pts_over_cost_neg": abs(pts_n) / cost_pts if cost_pts > 0 and not np.isnan(pts_n) else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--bars-glob", required=True)
    p.add_argument("--tune-csv", required=True)
    p.add_argument("--top", type=int, default=10, help="Rank-N combos to examine")
    p.add_argument("--cost-pts", type=float, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cost_pts = args.cost_pts if args.cost_pts is not None else DEFAULT_COST_PTS[args.instrument]
    top = _pick_top_combos(Path(args.tune_csv), n=args.top)
    print(f"[rank] top {args.top} combos by balance × |corr|:")
    print(top)

    bars = _load_bars(args.bars_glob)

    out_rows = []
    for combo in top.iter_rows(named=True):
        k_up = float(combo["k_up"]); k_dn = float(combo["k_dn"])
        T = int(combo["T"]); atr_w = int(combo["atr_window"])
        for y in IS_YEARS:
            bars_y = bars.filter(
                (pl.col("ts").dt.date() >= date(y, 1, 1))
                & (pl.col("ts").dt.date() <= date(y, 12, 31))
            )
            if bars_y.height == 0:
                print(f"[warn] {y}: no bars — skipping")
                continue
            stats = _year_stats(bars_y, k_up, k_dn, T, atr_w, cost_pts)
            stats.update(dict(
                instrument=args.instrument, year=y,
                k_up=k_up, k_dn=k_dn, T=T, atr_window=atr_w,
            ))
            out_rows.append(stats)

    out_df = pl.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)
    print(f"\n[stab] wrote {out_path}  ({out_df.height} rows)")

    # Per-combo regime-robustness summary: worst-year balance + corr across 2020-2023
    summary = (
        out_df.group_by(["k_up", "k_dn", "T", "atr_window"])
        .agg([
            pl.col("balance_score").min().alias("balance_min"),
            pl.col("balance_score").mean().alias("balance_mean"),
            pl.col("corr").min().alias("corr_min"),
            pl.col("corr").mean().alias("corr_mean"),
            pl.col("pts_over_cost_pos").min().alias("ptsOC_pos_min"),
            pl.col("pts_over_cost_neg").min().alias("ptsOC_neg_min"),
        ])
        .sort("balance_min", descending=True)
    )
    print("\n[stab] robustness summary (sorted by worst-year balance):")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
