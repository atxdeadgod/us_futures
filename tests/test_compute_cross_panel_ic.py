"""Smoke tests for compute_cross_panel_ic.py — IC + t-stat math + tie handling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import compute_cross_panel_ic as ic_mod  # noqa: E402


def test_ic_pearson_perfect_corr():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    y = 2 * x + 0.0  # perfectly linear
    ic, n = ic_mod._ic_pearson(x, y)
    assert abs(ic - 1.0) < 1e-9
    assert n == 1000


def test_ic_pearson_zero_corr():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    y = rng.standard_normal(5000)
    ic, n = ic_mod._ic_pearson(x, y)
    assert abs(ic) < 0.05  # noise around 0


def test_ic_spearman_handles_categorical_label_ties():
    """label ∈ {-1, 0, +1} with massive ties → Spearman should still rank-correlate."""
    rng = np.random.default_rng(2)
    feature = rng.standard_normal(3000)
    # Binarize feature into label, with noise so it's not perfect rank
    label = np.where(feature > 0.5, 1, np.where(feature < -0.5, -1, 0)).astype(float)
    ic, n = ic_mod._ic_spearman(feature, label)
    # Should be strongly positive (feature and label are monotonically related)
    assert ic > 0.6
    assert n == 3000


def test_ic_spearman_zero_corr():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(3000)
    label = rng.choice([-1, 0, 1], size=3000).astype(float)
    ic, n = ic_mod._ic_spearman(x, label)
    assert abs(ic) < 0.05


def test_tstat_formula():
    """t = ic * sqrt(n - 2) / sqrt(1 - ic^2). At ic=0.1, n=10000 → t ~= 10.05."""
    t = ic_mod._tstat(0.1, 10000)
    assert 9 < t < 11


def test_tstat_zero_for_zero_ic():
    assert ic_mod._tstat(0.0, 10000) == 0.0


def test_tstat_nan_when_unstable():
    # Tiny n
    assert np.isnan(ic_mod._tstat(0.5, 2))
    # |ic| = 1 would divide by zero
    assert np.isnan(ic_mod._tstat(1.0, 1000))
    assert np.isnan(ic_mod._tstat(float("nan"), 1000))


def test_select_feature_columns_filters_non_features():
    import polars as pl
    df = pl.DataFrame({
        "ts": [1, 2, 3], "open": [1.0, 2.0, 3.0], "label": [-1, 0, 1],
        "log_return": [0.01, -0.02, 0.03],
        "ofi_tc_z_w30": [0.5, -0.3, 0.1],
        "string_col": ["a", "b", "c"],  # non-numeric, should be skipped
    })
    feats = ic_mod._select_feature_columns(df)
    assert "log_return" in feats
    assert "ofi_tc_z_w30" in feats
    assert "ts" not in feats
    assert "open" not in feats
    assert "label" not in feats
    assert "string_col" not in feats


def test_ic_pearson_too_few_valid_returns_nan():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    ic, n = ic_mod._ic_pearson(x, y)
    # Only 3 valid pairs (positions 0, 1, 4); below 100 threshold → NaN
    assert np.isnan(ic)
    assert n == 3


def test_ic_handles_constant_feature():
    """All-same feature: stdev=0, corr undefined → NaN."""
    x = np.full(1000, 5.0)
    y = np.random.default_rng(4).standard_normal(1000)
    ic_p, _ = ic_mod._ic_pearson(x, y)
    ic_s, _ = ic_mod._ic_spearman(x, y)
    assert np.isnan(ic_p)
    assert np.isnan(ic_s)
