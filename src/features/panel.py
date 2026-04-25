"""Backward-compat facade — re-exports the split modules.

The original monolithic `panel.py` was split into focused modules during the
2026-04-25 refactor (see REFACTOR.md). This file remains as a thin import
shim so callers using `from src.features import panel; panel.X` continue to
work without modification. New code should import directly from:

    src.features.single_contract   — per-instrument feature attach passes
    src.features.external_sources  — VX / GEX attach onto a target
    src.features.cross_sectional   — multi-instrument joins, ranks, composites
    src.features.labeling          — V1 label params + assemble_target_panel
"""
from __future__ import annotations

from .cross_sectional import (
    ALL_INSTRUMENTS,
    ASSET_CLASSES,
    TRADING_INSTRUMENTS,
    attach_cross_asset_composites,
    attach_cross_sectional_ranks,
    build_wide_cross_asset_frame,
)
from .external_sources import (
    attach_gex_for_target,
    attach_vx_features,
)
from .labeling import (
    V1_COST_PTS,
    V1_LABEL_PARAMS,
    assemble_target_panel,
)
from .single_contract import (
    BASE_VALUE_COLS,
    attach_base_microstructure_features,
    attach_engine_features,
    attach_l2_deep_features,
    attach_pattern_features,
    attach_phase_e_features,
    attach_ts_normalizations,
    build_per_instrument_features,
)

__all__ = [
    # single_contract
    "BASE_VALUE_COLS",
    "attach_base_microstructure_features",
    "attach_engine_features",
    "attach_l2_deep_features",
    "attach_pattern_features",
    "attach_phase_e_features",
    "attach_ts_normalizations",
    "build_per_instrument_features",
    # external_sources
    "attach_gex_for_target",
    "attach_vx_features",
    # cross_sectional
    "ALL_INSTRUMENTS",
    "ASSET_CLASSES",
    "TRADING_INSTRUMENTS",
    "attach_cross_asset_composites",
    "attach_cross_sectional_ranks",
    "build_wide_cross_asset_frame",
    # labeling
    "V1_COST_PTS",
    "V1_LABEL_PARAMS",
    "assemble_target_panel",
]
