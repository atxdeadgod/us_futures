"""V1 label parameters and final per-target panel assembly.

The triple-barrier labeling primitive lives in `src/labels/triple_barrier.py`.
This module owns:
    - V1_LABEL_PARAMS / V1_COST_PTS: locked V1 hyperparameters per trading instrument
      (from LABELING_V1_SUMMARY.md tuning runs)
    - assemble_target_panel: applies labels to a per-target feature frame, optionally
      left-joining a wide cross-sectional frame, drops warmup/halt-truncated rows.
"""
from __future__ import annotations

import polars as pl

from ..labels.triple_barrier import triple_barrier_labels


# ---------------------------------------------------------------------------
# V1 locked label params per trading instrument (from LABELING_V1_SUMMARY.md)
# ---------------------------------------------------------------------------

V1_LABEL_PARAMS: dict[str, dict] = {
    "ES":  {"k_up": 1.25, "k_dn": 1.25, "T": 8, "lookback_days": 150},
    "NQ":  {"k_up": 1.25, "k_dn": 1.25, "T": 8, "lookback_days": 180},
    "RTY": {"k_up": 1.00, "k_dn": 1.00, "T": 4, "lookback_days": 180},
    "YM":  {"k_up": 1.25, "k_dn": 1.25, "T": 8, "lookback_days": 150},
}

# Round-trip cost (instrument points; spread + commission + slippage estimates)
V1_COST_PTS: dict[str, float] = {"ES": 0.50, "NQ": 1.50, "RTY": 0.30, "YM": 3.00}


def assemble_target_panel(
    target: str,
    target_bars_with_features: pl.DataFrame,
    wide_cross_asset: pl.DataFrame | None = None,
    label_params: dict | None = None,
    cost_pts: float | None = None,
    halt_mode: str = "truncate",
    min_effective_T: int = 5,
    partition_minutes: int = 15,
    bar_minutes: int = 15,
    drop_invalid: bool = True,
) -> pl.DataFrame:
    """Final per-target feature panel.

    Workflow:
      1. Start with target's own bars-with-features (per-instrument pipeline already applied,
         optionally with overnight + L2 features attached upstream).
      2. Left-join the wide cross-asset frame on `ts`, dropping the wide frame's
         `{target}_*` columns (already in target's bars).
      3. Apply triple_barrier_labels with V1 locked params for `target`.
      4. Filter to valid rows (atr + realized_ret finite) — drops warmup +
         halt-truncated bars where effective T < min_effective_T.

    Args:
        target: trading instrument key (must be in V1_LABEL_PARAMS)
        target_bars_with_features: target's Phase A (or A+B) bars with base
            features + normalizations + (optionally) overnight + L2 features
        wide_cross_asset: output of cross-sectional pipeline; if None, only
            target's own features go into the panel
        label_params: V1 label params for target. If None, uses V1_LABEL_PARAMS[target].
        cost_pts: round-trip cost. If None, uses V1_COST_PTS[target].
        halt_mode, min_effective_T, partition_minutes: V1 architecture defaults
            from LABELING_V1_SUMMARY.md (truncate, 5, 15)
        drop_invalid: filter out rows where atr or realized_ret is non-finite
            (warmup, halt-truncated, etc.). Default True.

    Returns:
        Per-bar feature panel with `label`, `realized_ret`, `realized_ret_pts`,
        `hit_offset`, `halt_truncated`, `atr` columns appended.
    """
    if label_params is None:
        if target not in V1_LABEL_PARAMS:
            raise ValueError(f"No V1_LABEL_PARAMS for target={target!r}; provide label_params explicitly")
        label_params = V1_LABEL_PARAMS[target]

    df = target_bars_with_features

    if wide_cross_asset is not None:
        target_prefix_cols = [c for c in wide_cross_asset.columns if c.startswith(f"{target}_")]
        wide_to_join = wide_cross_asset.drop(target_prefix_cols)
        already = set(df.columns)
        wide_unique_cols = ["ts"] + [c for c in wide_to_join.columns if c != "ts" and c not in already]
        wide_subset = wide_to_join.select(wide_unique_cols)
        df = df.join(wide_subset, on="ts", how="left")

    labeled = triple_barrier_labels(
        df,
        k_up=label_params["k_up"],
        k_dn=label_params["k_dn"],
        T=label_params["T"],
        atr_window=label_params.get("lookback_days", 60),
        atr_mode="time_conditional",
        lookback_days=label_params["lookback_days"],
        bar_minutes=bar_minutes,
        partition_minutes=partition_minutes,
        halt_aware=True,
        halt_mode=halt_mode,
        min_effective_T=min_effective_T,
    )

    if drop_invalid:
        labeled = labeled.filter(
            pl.col("atr").is_finite() & pl.col("realized_ret").is_finite()
        )

    return labeled
