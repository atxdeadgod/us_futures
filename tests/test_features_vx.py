"""Tests for src/features/vx.py — VX regime features."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import vx


def _mk_vx_panel(n: int = 30, contango: bool = True):
    """Build a VX1/VX2/VX3 wide panel. Contango: VX1 < VX2 < VX3."""
    vx1_mid = [15.0] * n
    vx2_mid = [16.0 if contango else 14.0] * n
    vx3_mid = [16.5 if contango else 13.5] * n
    cols = {}
    for prefix, mid in [("VX1", vx1_mid), ("VX2", vx2_mid), ("VX3", vx3_mid)]:
        cols[f"{prefix}_bid_px_L1"] = [m - 0.05 for m in mid]
        cols[f"{prefix}_ask_px_L1"] = [m + 0.05 for m in mid]
        cols[f"{prefix}_bid_sz_L1"] = [100] * n
        cols[f"{prefix}_ask_sz_L1"] = [100] * n
        # Pad levels 2..10 with zeros (not used in these tests)
        for k in range(2, 11):
            cols[f"{prefix}_bid_px_L{k}"] = [0.0] * n
            cols[f"{prefix}_ask_px_L{k}"] = [0.0] * n
            cols[f"{prefix}_bid_sz_L{k}"] = [0] * n
            cols[f"{prefix}_ask_sz_L{k}"] = [0] * n
    return pl.DataFrame(cols)


def test_vx_mid_simple():
    df = _mk_vx_panel()
    df = df.with_columns(vx.vx_mid("VX1").alias("m"))
    # (15 - 0.05 + 15 + 0.05) / 2 = 15.0
    assert abs(df["m"][0] - 15.0) < 1e-9


def test_vx_calendar_spread_contango():
    """VX1 < VX2 → spread negative (contango)."""
    df = _mk_vx_panel(contango=True)
    df = df.with_columns(vx.vx_calendar_spread("VX1", "VX2").alias("sp"))
    # 15 − 16 = -1
    assert abs(df["sp"][0] - (-1.0)) < 1e-9


def test_vx_calendar_spread_backwardation():
    df = _mk_vx_panel(contango=False)
    df = df.with_columns(vx.vx_calendar_spread("VX1", "VX2").alias("sp"))
    # 15 − 14 = +1 (backwardation)
    assert abs(df["sp"][0] - 1.0) < 1e-9


def test_vx_calendar_ratio():
    df = _mk_vx_panel(contango=True)
    df = df.with_columns(vx.vx_calendar_ratio("VX1", "VX2").alias("r"))
    # 15 / 16 ≈ 0.9375
    assert abs(df["r"][0] - (15/16)) < 1e-6


def test_vx_term_curvature_contango():
    df = _mk_vx_panel(contango=True)  # VX1=15, VX2=16, VX3=16.5
    df = df.with_columns(vx.vx_term_curvature().alias("c"))
    # 16.5 - 2*16 + 15 = -0.5 (concave down — typical vol curve)
    assert abs(df["c"][0] - (-0.5)) < 1e-9


def test_vx_term_curvature_backwardation():
    df = _mk_vx_panel(contango=False)  # VX1=15, VX2=14, VX3=13.5
    df = df.with_columns(vx.vx_term_curvature().alias("c"))
    # 13.5 - 2*14 + 15 = +0.5 (concave up — inverted curve)
    assert abs(df["c"][0] - 0.5) < 1e-9


def test_vx_zscore_constant_series():
    """Constant mid → zero std → z-score near 0 (with EPS)."""
    df = _mk_vx_panel(n=60)
    df = df.with_columns([
        vx.vx_mid("VX1").alias("m"),
    ])
    df = df.with_columns(vx.vx_zscore(pl.col("m"), window=20).alias("z"))
    # With a constant mid, rolling std = 0 → z = 0 / EPS = 0
    valid = df["z"].drop_nulls().to_list()
    assert all(abs(v) < 1e-3 for v in valid)


def test_vx_spread_zscore_returns_expr():
    """Returns a polars expression — plug into with_columns and no crash."""
    df = _mk_vx_panel(n=30)
    df = df.with_columns(vx.vx_spread_zscore("VX1", depth=1, window=10).alias("z"))
    # Constant spread of 0.1 → z should be 0 (after warmup)
    valid = df["z"].drop_nulls().to_list()
    assert all(abs(v) < 1e-3 for v in valid)


def test_vx_ofi_works_with_prefix():
    """vx_ofi_weighted reuses deep_ofi with prefix — should produce a column without error."""
    df = _mk_vx_panel(n=10)
    df = df.with_columns(vx.vx_ofi_weighted("VX1", max_depth=1, decay=0.0).alias("ofi"))
    # Static book → all OFIs should be 0 (no price or size change)
    valid = df["ofi"].drop_nulls().to_list()
    # First row is null (shift), rest should be 0
    assert all(abs(v) < 1e-9 for v in valid)
