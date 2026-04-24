"""Tests for src/features/bar_cross.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import bar_cross


def test_bar_lead_lag_return_shift():
    """Leader returns shifted by lag end up in the right row position."""
    df = pl.DataFrame({"lead": [100.0, 101.0, 102.01, 103.03, 104.06]})
    df = df.with_columns(bar_cross.bar_lead_lag_return(pl.col("lead"), lag=1).alias("lag1"))
    # Return series: log(101/100), log(102.01/101), log(103.03/102.01), log(104.06/103.03)
    # After shift(1): [nan, nan, log(101/100), log(102.01/101), log(103.03/102.01)]
    expected = [
        None, None,
        np.log(101/100),
        np.log(102.01/101),
        np.log(103.03/102.01),
    ]
    got = df["lag1"].to_list()
    assert got[0] is None
    assert got[1] is None
    assert abs(got[2] - expected[2]) < 1e-9
    assert abs(got[3] - expected[3]) < 1e-9


def test_return_dispersion_zero_on_locked():
    """If all peers have identical returns, dispersion = 0 (after warmup)."""
    n = 30
    price = [100.0 * (1.01 ** i) for i in range(n)]
    df = pl.DataFrame({"a": price, "b": price, "c": price})
    df = df.with_columns(
        bar_cross.return_dispersion(["a", "b", "c"], window=5).alias("disp")
    )
    # After warmup, dispersion should be ~0 (all same prices)
    valid = df["disp"].drop_nulls().to_list()
    assert all(abs(v) < 1e-9 for v in valid)


def test_return_dispersion_positive_on_random():
    rng = np.random.default_rng(0)
    n = 200
    prices = [100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01 + 0.0005)) for _ in range(4)]
    df = pl.DataFrame({f"p{i}": prices[i].tolist() for i in range(4)})
    df = df.with_columns(
        bar_cross.return_dispersion([f"p{i}" for i in range(4)], window=20).alias("disp")
    )
    assert df["disp"].drop_nulls().mean() > 0


def test_breadth_all_positive():
    """All peers trending up → breadth = 1.0 after warmup."""
    df = pl.DataFrame({
        "a": [100.0, 101.0, 102.0, 103.0],
        "b": [50.0, 51.0, 52.0, 53.0],
        "c": [200.0, 210.0, 220.0, 230.0],
    })
    df = df.with_columns(bar_cross.breadth(["a", "b", "c"]).alias("br"))
    # Row 0 is warmup (no prior prices for any peer). Row 1..3 should all be 1.0
    valid = df["br"].to_list()[1:]
    assert all(abs(v - 1.0) < 1e-9 for v in valid)


def test_breadth_half_half():
    """Half peers up, half down → breadth = 0.5 after warmup."""
    df = pl.DataFrame({
        "a": [100.0, 101.0, 102.0],  # up
        "b": [100.0, 99.0, 98.0],     # down
        "c": [100.0, 101.0, 102.0],  # up
        "d": [100.0, 99.0, 98.0],     # down
    })
    df = df.with_columns(bar_cross.breadth(["a", "b", "c", "d"]).alias("br"))
    valid = df["br"].to_list()[1:]
    assert all(abs(v - 0.5) < 1e-9 for v in valid)


def test_leader_laggard_spread():
    """Wider cross-sectional return range → bigger leader-laggard spread."""
    df = pl.DataFrame({
        "a": [100.0, 105.0],   # +5% return
        "b": [100.0, 95.0],    # -5% return
        "c": [100.0, 100.0],   # 0
    })
    df = df.with_columns(bar_cross.leader_laggard_spread(["a", "b", "c"]).alias("sp"))
    # Row 1: max return = ln(105/100), min = ln(95/100)
    assert abs(df["sp"][1] - (np.log(105/100) - np.log(95/100))) < 1e-9


def test_return_concentration_uniform():
    """Uniform |returns| → concentration = 1/N."""
    df = pl.DataFrame({
        "a": [100.0, 101.0],   # +log(1.01)
        "b": [100.0, 99.0],    # -|log(1.01)| ≈ same magnitude
        "c": [100.0, 101.0],   # +log(1.01)
        "d": [100.0, 99.0],    # -|log(1.01)| approx
    })
    df = df.with_columns(bar_cross.return_concentration(["a", "b", "c", "d"]).alias("hhi"))
    # With nearly equal magnitude returns, concentration ≈ 1/4
    val = df["hhi"][1]
    assert 0.22 < val < 0.28  # ~0.25, small deviation because 99→100 isn't exactly -log(1.01)


def test_return_concentration_dominated():
    """One peer moves much more than others → concentration > 0.5."""
    df = pl.DataFrame({
        "a": [100.0, 150.0],  # +50% huge return
        "b": [100.0, 100.5],
        "c": [100.0, 100.3],
        "d": [100.0, 100.2],
    })
    df = df.with_columns(bar_cross.return_concentration(["a", "b", "c", "d"]).alias("hhi"))
    val = df["hhi"][1]
    assert val > 0.7  # A dominates, concentration high


def test_cross_sectional_return_zscore():
    """Target return well above peer mean → positive z-score."""
    df = pl.DataFrame({
        "tgt": [100.0, 110.0],  # huge jump
        "p1": [100.0, 101.0],
        "p2": [100.0, 100.5],
        "p3": [100.0, 100.8],
    })
    df = df.with_columns(
        bar_cross.cross_sectional_return_zscore("tgt", ["p1", "p2", "p3"]).alias("z")
    )
    assert df["z"][1] > 5  # target way above peer mean → large positive z
