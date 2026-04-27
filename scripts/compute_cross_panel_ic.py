"""IC + t-stat dashboard for cross panel features (2020-2023 IS, OOS withheld).

For each (target, feature) pair, computes correlation against both:
  - `label` (V1 triple-barrier {-1, 0, +1}, ordinal)  — Spearman is appropriate
  - `realized_ret` (continuous)                         — Spearman AND Pearson

We compute IC against `label` because that's the actual trading decision
the model will produce. We also compute against `realized_ret` because the
continuous target carries more granular signal and helps distinguish
"weakly predictive of magnitude" from "barrier-hit lottery".

t-stat = ic * sqrt(N - 2) / sqrt(1 - ic^2), df = N - 2.
A t-stat above ±2.5 (2-tailed p < 0.012) is the rough significance threshold
at typical N. With ~30-40K samples per target panel, even small ICs (~0.02)
become highly significant — the goal is RELATIVE ranking, not p-value pass/fail.

The `_tc_residual` variant subtracts the hour-of-day mean from the target
to remove session-conditional effects (a feature that tracks "RTH bars
have higher vol" gets discounted vs one that predicts within-session).

In-sample window: 2020-2023 ONLY. Excluding 2024 (the OOS year) avoids
lookahead bias when ILP picks features for the model.

Usage:
    python scripts/compute_cross_panel_ic.py \\
        --target ES \\
        --cross-root /N/.../features/cross \\
        --out-root  /N/.../ic_dashboard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


IS_YEARS = (2020, 2021, 2022, 2023)
OOS_YEARS = (2024,)

# Cols never treated as features (target outputs, raw bar-build inputs, etc.)
NON_FEATURE_COLS = {
    # Identity / time
    "ts", "root", "expiry",
    # Raw bar inputs (model uses derived; raw would be redundant + leak)
    "open", "high", "low", "close", "volume", "dollar_volume",
    "buys_qty", "sells_qty", "trades_count", "unclassified_count",
    "implied_volume", "implied_buys", "implied_sells",
    "bid_close", "ask_close", "mid_close", "spread_abs_close",
    "spread_mean_sub", "spread_std_sub", "spread_max_sub", "spread_min_sub",
    "cvd_globex", "cvd_rth", "bars_since_rth_reset",
    "is_session_warm",
    # Raw L1-L10 book columns (always-on for trading 4)
    *(f"{side}_{kind}_L{k}" for side in ("bid", "ask")
      for kind in ("px", "sz", "ord") for k in range(1, 11)),
    "book_ts_close", "book_ts_close_bid", "book_ts_close_ask",
    # Targets — predicting these would be circular
    "label", "realized_ret", "realized_ret_pts", "hit_offset", "atr",
    "halt_truncated",
}


def _select_feature_columns(panel: pl.DataFrame) -> list[str]:
    """Numeric columns not on the deny list."""
    return [
        c for c, dt in zip(panel.columns, panel.dtypes)
        if c not in NON_FEATURE_COLS and dt.is_numeric()
    ]


def _ic_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Pearson correlation + n_samples after dropping non-finite."""
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 100:
        return float("nan"), n
    xv, yv = x[valid], y[valid]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan"), n
    return float(np.corrcoef(xv, yv)[0, 1]), n


def _ic_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman rank correlation = Pearson on average-ranked values.

    Uses scipy.stats.rankdata(method='average') so tied values get their
    mean rank — critical when one operand is a categorical target like
    `label` ∈ {-1, 0, +1} where ~60-70% of samples are tied.
    """
    from scipy.stats import rankdata
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 100:
        return float("nan"), n
    xv, yv = x[valid], y[valid]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan"), n
    rx = rankdata(xv, method="average")
    ry = rankdata(yv, method="average")
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _tstat(ic: float, n: int) -> float:
    """Two-tailed t-stat from correlation. df = n - 2."""
    if not np.isfinite(ic) or n < 3 or abs(ic) >= 1:
        return float("nan")
    return ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2)


def _tc_residual(panel: pl.DataFrame, target_col: str, hour_col: str = "hour_et") -> np.ndarray:
    """target − rolling-mean(target | hour_of_day). Removes session-conditional bias."""
    if hour_col not in panel.columns:
        df = panel.with_columns(pl.col("ts").dt.hour().alias("_hour"))
        hour_col = "_hour"
    else:
        df = panel
    df = df.with_columns(pl.col(target_col).mean().over(hour_col).alias("_hour_mean"))
    return (df[target_col] - df["_hour_mean"]).to_numpy()


def _load_target_panels(cross_root: Path, target: str, years: tuple[int, ...]) -> pl.DataFrame | None:
    """Concat cross panels across years. Returns None if no parquets found."""
    files = []
    for yr in years:
        path = cross_root / f"{target}_{yr}.parquet"
        if path.exists():
            files.append(path)
    if not files:
        return None
    # diagonal_relaxed handles col-set differences across years (which we expect
    # — different feature counts post-Phase-E retrofit, etc.)
    return pl.concat(
        [pl.read_parquet(p) for p in files], how="diagonal_relaxed"
    ).sort("ts")


def _load_per_year_panels(
    cross_root: Path, target: str, years: tuple[int, ...]
) -> dict[int, pl.DataFrame]:
    """Return {year: panel_df} for every year that has a parquet."""
    out: dict[int, pl.DataFrame] = {}
    for yr in years:
        path = cross_root / f"{target}_{yr}.parquet"
        if path.exists():
            out[yr] = pl.read_parquet(path).sort("ts")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--cross-root", required=True,
                   help="Root containing cross/{TARGET}_{YEAR}.parquet")
    p.add_argument("--out-root", required=True,
                   help="Output dir for {target}_ic_2020_2023.csv")
    args = p.parse_args()

    cross_root = Path(args.cross_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load per-year panels (used both individually for stability AND pooled
    # for the aggregate IC).
    per_year_panels = _load_per_year_panels(cross_root, args.target, IS_YEARS)
    if not per_year_panels:
        print(f"[err] no {args.target} cross panels found under {cross_root} for {IS_YEARS}",
              file=sys.stderr)
        return 1
    years_loaded = sorted(per_year_panels.keys())
    print(f"[load] {args.target} IS years loaded: {years_loaded}")
    for yr, p_df in per_year_panels.items():
        print(f"  {yr}: rows={p_df.height:,}  cols={len(p_df.columns)}")

    # Pooled panel = concat across years for the aggregate IC
    panel = pl.concat(
        list(per_year_panels.values()), how="diagonal_relaxed"
    ).sort("ts")
    print(f"[load] pooled IS panel: rows={panel.height:,}  cols={len(panel.columns)}")

    for required in ("label", "realized_ret"):
        if required not in panel.columns:
            print(f"[err] required target col {required!r} not in panel", file=sys.stderr)
            return 1

    feature_cols = _select_feature_columns(panel)
    print(f"[ic] feature columns: {len(feature_cols)}")

    # Pre-extract target arrays (pooled + per-year)
    label_arr = panel["label"].cast(pl.Float64).to_numpy()
    ret_arr = panel["realized_ret"].to_numpy()
    ret_tc_resid = _tc_residual(panel, "realized_ret")
    per_year_label = {
        yr: per_year_panels[yr]["label"].cast(pl.Float64).to_numpy()
        for yr in years_loaded
    }

    print(f"[ic] computing pooled IC + per-year stability for {len(feature_cols)} features...")
    rows = []
    for i, feat in enumerate(feature_cols):
        try:
            arr = panel[feat].cast(pl.Float64, strict=False).to_numpy()
        except Exception:
            continue
        ic_label_sp, n_label = _ic_spearman(arr, label_arr)
        ic_ret_sp, n_ret = _ic_spearman(arr, ret_arr)
        ic_ret_pe, _ = _ic_pearson(arr, ret_arr)
        ic_ret_tc_pe, _ = _ic_pearson(arr, ret_tc_resid)

        # Per-year IC vs label (Spearman). Stability tells us if the pooled
        # IC is a regime-dependent fluke or a real effect.
        per_year_ic: dict[int, float] = {}
        for yr in years_loaded:
            if feat not in per_year_panels[yr].columns:
                per_year_ic[yr] = float("nan")
                continue
            yr_arr = per_year_panels[yr][feat].cast(pl.Float64, strict=False).to_numpy()
            yr_ic, _ = _ic_spearman(yr_arr, per_year_label[yr])
            per_year_ic[yr] = yr_ic

        finite_yr_ics = [v for v in per_year_ic.values() if np.isfinite(v)]
        if finite_yr_ics:
            ic_yr_mean = float(np.mean(finite_yr_ics))
            ic_yr_std = float(np.std(finite_yr_ics, ddof=0))
            ic_yr_min = float(min(finite_yr_ics))
            ic_yr_max = float(max(finite_yr_ics))
            # Sign-stability: fraction of finite years with same sign as the pooled IC
            if np.isfinite(ic_label_sp) and ic_label_sp != 0:
                sign_pooled = np.sign(ic_label_sp)
                sign_match = sum(1 for v in finite_yr_ics if np.sign(v) == sign_pooled)
                sign_stable = sign_match / len(finite_yr_ics)
            else:
                sign_stable = float("nan")
        else:
            ic_yr_mean = ic_yr_std = ic_yr_min = ic_yr_max = sign_stable = float("nan")

        row = {
            "feature": feat,
            "n": n_label,
            # Pooled (aggregate over all years)
            "ic_label_spearman": ic_label_sp,
            "tstat_label_spearman": _tstat(ic_label_sp, n_label),
            "ic_ret_spearman": ic_ret_sp,
            "tstat_ret_spearman": _tstat(ic_ret_sp, n_ret),
            "ic_ret_pearson": ic_ret_pe,
            "tstat_ret_pearson": _tstat(ic_ret_pe, n_ret),
            "ic_ret_tc_residual_pearson": ic_ret_tc_pe,
            "tstat_ret_tc_residual_pearson": _tstat(ic_ret_tc_pe, n_ret),
            # Per-year IC (stability check)
            "ic_label_yr_mean": ic_yr_mean,
            "ic_label_yr_std": ic_yr_std,
            "ic_label_yr_min": ic_yr_min,
            "ic_label_yr_max": ic_yr_max,
            "ic_label_sign_stable_frac": sign_stable,
        }
        for yr in years_loaded:
            row[f"ic_label_{yr}"] = per_year_ic[yr]
        # Pre-computed |IC| for sorting
        row["abs_ic_label_spearman"] = abs(ic_label_sp) if np.isfinite(ic_label_sp) else float("nan")
        row["abs_ic_ret_spearman"] = abs(ic_ret_sp) if np.isfinite(ic_ret_sp) else float("nan")
        row["abs_ic_ret_tc_residual"] = abs(ic_ret_tc_pe) if np.isfinite(ic_ret_tc_pe) else float("nan")
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(feature_cols)}")

    # polars `nulls_last=True` only handles Polars Null, not float NaN. Convert
    # NaN → null on the sort key so they get pushed to the bottom.
    out = (
        pl.DataFrame(rows)
        .with_columns(
            pl.when(pl.col("abs_ic_label_spearman").is_finite())
            .then(pl.col("abs_ic_label_spearman"))
            .otherwise(None)
            .alias("abs_ic_label_spearman")
        )
        .sort("abs_ic_label_spearman", descending=True, nulls_last=True)
    )
    out_path = out_root / f"{args.target}_ic_2020_2023.csv"
    out.write_csv(out_path)
    print(f"[done] wrote {out_path}  ({out.height} features ranked)")

    print("\n[top 25 by |IC vs label| (Spearman)]")
    print(out.head(25).select([
        "feature", "n",
        "ic_label_spearman", "tstat_label_spearman",
        "ic_ret_spearman", "ic_ret_tc_residual_pearson",
    ]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
