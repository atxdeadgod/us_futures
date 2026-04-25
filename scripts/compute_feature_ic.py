"""Compute IC of each feature in a panel against the labels.

Three IC variants per feature:

  1. ic_raw:                pearson(feature, realized_ret)
  2. ic_tc_residual:        pearson(feature, realized_ret − hour_of_day_mean)
  3. ic_naive_residual:     pearson(feature, realized_ret − naive_OLS_baseline_prediction)

The TC-residual and naive-residual variants filter out features that look good
on raw IC but only correlate with cheap baselines. Features with high IC across
all three are true candidates for the model.

Output: per-feature CSV ranked by `abs_ic_tc_residual` by default.

Usage:
    python scripts/compute_feature_ic.py \
        --panel /N/.../feature_panels/ES_features_panel_2020_2023.parquet \
        --out   /N/.../ic_dashboard/ES_ic_2020_2023.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


# Columns to EXCLUDE from feature evaluation. Includes raw bar columns,
# session flags (already-engineered binary features stay), label outputs,
# and bookkeeping cols.
NON_FEATURE_COLS = {
    # Identity / time
    "ts", "root", "expiry",
    # Raw OHLC + volume + L1 close (these are model INPUTS to derived features,
    # not features themselves — including would over-state IC of trivially-correlated cols)
    "open", "high", "low", "close", "volume", "dollar_volume",
    "buys_qty", "sells_qty", "trades_count", "unclassified_count",
    "implied_volume", "implied_buys", "implied_sells",
    "bid_close", "ask_close", "mid_close", "spread_abs_close",
    "spread_mean_sub", "spread_std_sub", "spread_max_sub", "spread_min_sub",
    "cvd_globex", "cvd_rth", "bars_since_rth_reset",
    "is_session_warm",
    # Raw L1-L10 book columns (when present from Phase A+B)
    *(f"{side}_{kind}_L{k}" for side in ("bid", "ask") for kind in ("px", "sz", "ord") for k in range(1, 11)),
    "book_ts_close", "book_ts_close_bid", "book_ts_close_ask",
    # Label outputs (we predict these — leakage if used as features)
    "label", "realized_ret", "realized_ret_pts", "hit_offset", "atr",
    "halt_truncated",
    # Session flag interpretation depends on use; keep the flags as features.
    # hour_et stays as feature; minute_of_day_sin/cos stay as features.
}


def _select_feature_columns(panel: pl.DataFrame) -> list[str]:
    """Return list of feature columns (everything that's numeric and not on the deny list)."""
    feats = []
    for c, dt in zip(panel.columns, panel.dtypes):
        if c in NON_FEATURE_COLS:
            continue
        if dt.is_numeric():
            feats.append(c)
    return feats


def _ic(feature: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation between feature and target, ignoring NaN."""
    valid = np.isfinite(feature) & np.isfinite(target)
    if valid.sum() < 100:
        return float("nan")
    f, t = feature[valid], target[valid]
    if np.std(f) == 0 or np.std(t) == 0:
        return float("nan")
    return float(np.corrcoef(f, t)[0, 1])


def _compute_tc_residual(
    panel: pl.DataFrame, target_col: str = "realized_ret", hour_col: str = "hour_et",
) -> np.ndarray:
    """target - mean(target | hour_of_day). Hour-of-day mean computed over the panel."""
    if hour_col not in panel.columns:
        # Fallback: use ts hour
        df = panel.with_columns(pl.col("ts").dt.hour().alias("_hour_for_resid"))
        hour_col = "_hour_for_resid"
    else:
        df = panel
    df = df.with_columns(pl.col(target_col).mean().over(hour_col).alias("_hour_mean"))
    return (df[target_col] - df["_hour_mean"]).to_numpy()


def _compute_naive_baseline_residual(
    panel: pl.DataFrame, target_col: str = "realized_ret",
    predictor_cols: tuple[str, ...] = ("log_return", "realized_vol_w20"),
) -> np.ndarray:
    """target − OLS(target ~ a + b*pred1 + c*pred2 + ...).

    OLS fit on the available rows (where target + all predictors are finite).
    Predictions emitted on every row (NaN where any predictor is NaN).
    Residual = target − prediction.
    """
    available_preds = [c for c in predictor_cols if c in panel.columns]
    if not available_preds:
        # No predictors found → residual = target − mean
        y = panel[target_col].to_numpy()
        valid = np.isfinite(y)
        if valid.sum() == 0:
            return y
        mu = float(np.mean(y[valid]))
        return y - mu
    y = panel[target_col].to_numpy()
    X = np.column_stack([np.ones(len(panel))] + [panel[c].to_numpy() for c in available_preds])
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if valid.sum() < len(available_preds) + 10:
        return y - np.nanmean(y)
    beta, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
    pred = X @ beta
    return y - pred


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True, help="Input feature panel parquet")
    p.add_argument("--out", required=True, help="Output ranked IC CSV")
    p.add_argument("--target-col", default="realized_ret",
                   help="Target column for IC computation (default: realized_ret)")
    p.add_argument("--hour-col", default="hour_et",
                   help="Hour-of-day column for TC-residual computation (default: hour_et)")
    p.add_argument("--baseline-predictors", default="log_return,realized_vol_w20",
                   help="Comma-separated columns for naive-baseline residualization")
    p.add_argument("--rank-by", default="abs_ic_tc_residual",
                   choices=["abs_ic_raw", "abs_ic_tc_residual", "abs_ic_naive_residual"],
                   help="Column to sort the output CSV by (descending)")
    args = p.parse_args()

    panel = pl.read_parquet(args.panel)
    print(f"[ic] panel: {panel.height:,} rows × {panel.width} cols")

    if args.target_col not in panel.columns:
        raise SystemExit(f"target_col={args.target_col!r} not found in panel")

    feature_cols = _select_feature_columns(panel)
    print(f"[ic] feature columns: {len(feature_cols)}")

    target_raw = panel[args.target_col].to_numpy()
    target_tc_resid = _compute_tc_residual(panel, args.target_col, args.hour_col)
    pred_cols = tuple(c.strip() for c in args.baseline_predictors.split(",") if c.strip())
    target_naive_resid = _compute_naive_baseline_residual(panel, args.target_col, pred_cols)

    print(f"[ic] computing IC variants for {len(feature_cols)} features...")
    rows = []
    for feat in feature_cols:
        try:
            arr = panel[feat].cast(pl.Float64, strict=False).to_numpy()
        except Exception:
            continue
        ic_raw = _ic(arr, target_raw)
        ic_tc = _ic(arr, target_tc_resid)
        ic_nv = _ic(arr, target_naive_resid)
        rows.append({
            "feature": feat,
            "ic_raw": ic_raw,
            "ic_tc_residual": ic_tc,
            "ic_naive_residual": ic_nv,
            "abs_ic_raw": abs(ic_raw) if np.isfinite(ic_raw) else float("nan"),
            "abs_ic_tc_residual": abs(ic_tc) if np.isfinite(ic_tc) else float("nan"),
            "abs_ic_naive_residual": abs(ic_nv) if np.isfinite(ic_nv) else float("nan"),
        })

    out_df = pl.DataFrame(rows).sort(args.rank_by, descending=True, nulls_last=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)
    print(f"[ic] wrote {out_path}  ({out_df.height} rows)")

    print("\n[ic] top 30 features by", args.rank_by)
    print(out_df.head(30).select(
        ["feature", "ic_raw", "ic_tc_residual", "ic_naive_residual"]
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
