"""ILP-style greedy feature selection with pairwise-correlation constraint.

Given an IC ranking (output of compute_feature_ic.py) and the panel itself
(for pairwise correlation computation), select up to `max_features` features
such that:

  1. Each selected feature has |IC| > min_ic
  2. No two selected features have |pairwise corr| >= max_corr (default 0.45)

Greedy implementation (1/2-approximation in worst case for MWIS, but in
practice near-optimal on sparsely-correlated feature sets):
  - Sort candidates by |IC| descending
  - Iterate: accept the next feature if it doesn't conflict with any
    already-accepted feature; otherwise skip
  - Stop when max_features reached

This matches the tognn_us approach (`stage2_ilp_selection.py`) but
implemented inline without an external ILP solver — fast and deterministic.

Usage:
    python scripts/ilp_select_features.py \
        --ic-csv /N/.../ic_dashboard/ES_ic_2020_2023.csv \
        --panel  /N/.../feature_panels/ES_features_panel_2020_2023.parquet \
        --out    /N/.../ilp_selection/ES_selected_features.csv \
        --rank-col abs_ic_tc_residual \
        --max-features 80 \
        --max-corr 0.45 \
        --min-ic 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def greedy_select(
    feature_names: list[str],
    ic_scores: list[float],
    corr_matrix: np.ndarray,
    max_features: int,
    max_corr: float,
) -> list[int]:
    """Greedy MWIS-like selection.

    Returns indices (into feature_names) of selected features, in selection order.
    `corr_matrix[i, j]` should hold pearson(feature_i, feature_j); NaN treated
    as no-conflict (e.g., for features that don't co-occur).

    Pre-condition: feature_names is already sorted by descending |ic_score|.
    """
    selected: list[int] = []
    n = len(feature_names)
    for i in range(n):
        if len(selected) >= max_features:
            break
        conflict = False
        for j in selected:
            r = corr_matrix[i, j]
            if not np.isnan(r) and abs(r) >= max_corr:
                conflict = True
                break
        if not conflict:
            selected.append(i)
    return selected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ic-csv", required=True, help="Output of compute_feature_ic.py")
    p.add_argument("--panel", required=True, help="Original panel parquet (for corr matrix)")
    p.add_argument("--out", required=True, help="Output: selected-features CSV")
    p.add_argument("--rank-col", default="abs_ic_tc_residual",
                   choices=["abs_ic_raw", "abs_ic_tc_residual", "abs_ic_naive_residual"],
                   help="IC column to rank/filter by")
    p.add_argument("--max-features", type=int, default=80,
                   help="Max number of features to select (default 80, matching tognn_us)")
    p.add_argument("--max-corr", type=float, default=0.45,
                   help="Pairwise correlation threshold (default 0.45)")
    p.add_argument("--min-ic", type=float, default=0.01,
                   help="Skip features below this |IC| threshold")
    args = p.parse_args()

    # Load + filter IC ranking. Defensive: drop null/NaN/non-finite rank values
    # (e.g., constant-zero feature columns produce NaN IC).
    ic_df = pl.read_csv(args.ic_csv).sort(args.rank_col, descending=True, nulls_last=True)
    ic_df = ic_df.filter(
        pl.col(args.rank_col).is_not_null()
        & pl.col(args.rank_col).is_finite()
        & (pl.col(args.rank_col) > args.min_ic)
    )
    print(f"[ilp] {ic_df.height} features pass min_ic={args.min_ic}")

    if ic_df.height == 0:
        raise SystemExit(f"No features above min_ic={args.min_ic}; lower the threshold or check IC CSV")

    feature_names = ic_df["feature"].to_list()
    ic_scores = ic_df[args.rank_col].to_list()

    # Load panel + extract feature columns
    panel = pl.read_parquet(args.panel)
    print(f"[ilp] panel: {panel.height:,} rows × {panel.width} cols")

    # Compute pairwise pearson correlation (use pandas for robust NaN handling)
    print(f"[ilp] computing {len(feature_names)}×{len(feature_names)} correlation matrix...")
    pd_df = panel.select(feature_names).to_pandas()
    corr = pd_df.corr(method="pearson", numeric_only=True).values

    # Greedy selection
    selected_idx = greedy_select(
        feature_names=feature_names,
        ic_scores=ic_scores,
        corr_matrix=corr,
        max_features=args.max_features,
        max_corr=args.max_corr,
    )
    selected_names = [feature_names[i] for i in selected_idx]
    print(f"[ilp] selected {len(selected_names)} features")

    # Write the IC rows for selected features (preserves their IC stats + ordering)
    out_df = ic_df.filter(pl.col("feature").is_in(selected_names))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)
    print(f"[ilp] wrote {out_path}  ({out_df.height} rows)")

    print("\n[ilp] selected features (top 30):")
    print(out_df.head(30).select(
        ["feature", "ic_raw", "ic_tc_residual", "ic_naive_residual"]
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
