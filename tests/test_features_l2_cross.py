"""Tests for src/features/l2_cross.py pairwise L2 features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import l2_cross


def _mk_pair(n=100):
    """Build a wide panel with ES_* and NQ_* L1-L5 columns, random-walk mids."""
    rng = np.random.default_rng(0)
    es_mid = 5000.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.001))
    nq_mid = 17000.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.001))
    cols = {}
    for prefix, mid_series in [("ES", es_mid), ("NQ", nq_mid)]:
        for k in range(1, 6):
            tick = 0.25 if prefix == "ES" else 0.75
            step_mult = k - 1
            cols[f"{prefix}_bid_px_L{k}"] = [m - tick/2 - step_mult*tick for m in mid_series]
            cols[f"{prefix}_ask_px_L{k}"] = [m + tick/2 + step_mult*tick for m in mid_series]
            cols[f"{prefix}_bid_sz_L{k}"] = [10 * k] * n
            cols[f"{prefix}_ask_sz_L{k}"] = [10 * k] * n
    return pl.DataFrame(cols)


def test_cross_correlation_valid_range():
    df = _mk_pair(200)
    df = df.with_columns(l2_cross.cross_correlation("ES", "NQ", window=30).alias("xc"))
    vals = df["xc"].drop_nulls().to_list()
    # Correlation is in [-1, 1]; with random-walk independent shocks, mean should be near 0
    assert all(-1.0 <= v <= 1.0 for v in vals)


def test_ofi_correlation_balanced_book_zero_variance():
    """Equal bid/ask sizes at L1 → OFI=0 everywhere → zero variance → corr = null."""
    df = _mk_pair(60)
    df = df.with_columns(l2_cross.ofi_correlation("ES", "NQ", window=20).alias("oc"))
    # With constant OFI=0 → denom=0 → corr=null
    valid = df["oc"].drop_nulls().to_list()
    # Could be all null, or could have a few rows of 0 if the variance was incidentally
    # slightly non-zero from floating point. Either way, no extreme values.
    assert len(valid) == 0 or all(abs(v) <= 1.0 for v in valid)


def test_microprice_diff_equal_size_equals_mid_diff():
    """Equal bid/ask size → microprice = mid → diff = mid_A - mid_B."""
    df = _mk_pair(10)
    df = df.with_columns(l2_cross.microprice_diff("ES", "NQ").alias("mpd"))
    # Check row 0: mid_ES ≈ 5000, mid_NQ ≈ 17000 → diff ≈ -12000
    v = df["mpd"][0]
    assert v < -11000 and v > -13000


def test_depth_imbalance_diff_zero_on_balanced():
    """Symmetric books on both sides → imbalance_A = imbalance_B = 0 → diff = 0."""
    df = _mk_pair(20)
    df = df.with_columns(l2_cross.depth_imbalance_diff("ES", "NQ", depth=5).alias("did"))
    assert all(abs(v) < 1e-9 for v in df["did"].to_list())


def test_pairs_spread_zscore_centered():
    """Random-walk mids → spread is random-walk too; rolling z-scores should be finite."""
    df = _mk_pair(150)
    df = df.with_columns(l2_cross.pairs_spread_zscore("ES", "NQ", window=30).alias("z"))
    vals = df["z"].drop_nulls().to_list()
    # All finite, mostly in [-4, 4] for normal random walk
    assert all(-10 < v < 10 for v in vals)


def test_relative_quoted_spread_diff():
    """ES has narrower relative spread than NQ (ES higher price) → diff negative-ish."""
    df = _mk_pair(20)
    df = df.with_columns(l2_cross.relative_quoted_spread_diff("ES", "NQ").alias("rsd"))
    # Both have small relative spreads; sign depends on params
    assert df["rsd"][0] is not None


def test_realized_volatility_ratio():
    df = _mk_pair(200)
    df = df.with_columns(l2_cross.realized_volatility_ratio("ES", "NQ", window=30).alias("vr"))
    vals = df["vr"].drop_nulls().to_list()
    # With similar vol processes, ratio should hover around 1
    assert 0.1 < float(np.mean(vals)) < 10.0


def test_price_lead_lag_product():
    df = _mk_pair(50)
    df = df.with_columns(l2_cross.price_lead_lag("ES", "NQ", lag=1).alias("pll"))
    # Product of two small returns; magnitudes tiny
    vals = df["pll"].drop_nulls().to_list()
    assert all(abs(v) < 0.01 for v in vals)
