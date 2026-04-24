"""Tests for src/features/patterns.py — Tier-7 stateful rolling features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import patterns


# ===========================================================================
# T7.01 Absorption score
# ===========================================================================

def test_absorption_score_positive_on_flow_without_move():
    """Sustained flow with ~zero return vol → high absorption score."""
    n = 50
    df = pl.DataFrame({
        "flow": [1000.0] * n,      # constant large dollar flow
        "retvol": [0.0001] * n,    # tiny vol
        "notional": [500_000.0] * n,
    })
    df = df.with_columns(
        patterns.absorption_score(
            pl.col("flow"), pl.col("retvol"), pl.col("notional"), window=20
        ).alias("abs")
    )
    valid = df["abs"].drop_nulls()
    assert valid[-1] > 0  # flow with no vol = absorption


# ===========================================================================
# T7.03 Volume-at-price concentration
# ===========================================================================

def test_volume_at_price_concentration_range():
    df = pl.DataFrame({"bar_vol": [100, 100, 100], "near_vol": [10, 50, 100]})
    df = df.with_columns(
        patterns.volume_at_price_concentration(pl.col("bar_vol"), pl.col("near_vol")).alias("c")
    )
    assert abs(df["c"][0] - 0.1) < 1e-9
    assert abs(df["c"][1] - 0.5) < 1e-9
    assert abs(df["c"][2] - 1.0) < 1e-9


# ===========================================================================
# T7.04 Breakout magnitude
# ===========================================================================

def test_breakout_magnitude_up_on_new_high():
    # 30 bars of high=100, then a bar at 105 with ATR=2 → magnitude = 5/2 = 2.5
    highs = [100.0] * 31 + [105.0]
    atr = [2.0] * 32
    df = pl.DataFrame({"h": highs, "atr": atr})
    df = df.with_columns(
        patterns.breakout_magnitude_up(pl.col("h"), pl.col("atr"), lookback_bars=30).alias("bm")
    )
    # At the 105 bar, prior 30-bar max = 100, current = 105 → (105-100)/2 = 2.5
    assert abs(df["bm"][-1] - 2.5) < 1e-6
    # Earlier bars with same high = 100: (100-100)/2 = 0 (no breakout)
    assert df["bm"][-2] == 0.0


def test_breakout_magnitude_down_on_new_low():
    lows = [100.0] * 31 + [95.0]
    atr = [2.0] * 32
    df = pl.DataFrame({"l": lows, "atr": atr})
    df = df.with_columns(
        patterns.breakout_magnitude_down(pl.col("l"), pl.col("atr"), lookback_bars=30).alias("bm")
    )
    assert abs(df["bm"][-1] - 2.5) < 1e-6


# ===========================================================================
# T7.05 Breakout reversal (Wyckoff upthrust / spring)
# ===========================================================================

def test_breakout_reversal_up_fires_on_bull_trap():
    """Bar pokes above prior high then closes well below → upthrust flag = 1."""
    highs = [100.0] * 31 + [103.0]     # new high
    closes = [99.0] * 31 + [98.5]      # but closed back below 100 by 1.5 (0.75 ATR)
    atr = [2.0] * 32
    df = pl.DataFrame({"h": highs, "c": closes, "atr": atr})
    df = df.with_columns(
        patterns.breakout_reversal_up(
            pl.col("h"), pl.col("c"), pl.col("atr"),
            lookback_bars=30, reversal_atr=0.5,
        ).alias("bru")
    )
    assert df["bru"][-1] == 1


def test_breakout_reversal_up_no_fire_on_clean_breakout():
    """Bar breaks high AND closes above → NOT a reversal."""
    highs = [100.0] * 31 + [103.0]
    closes = [99.0] * 31 + [102.5]  # closed above 100
    atr = [2.0] * 32
    df = pl.DataFrame({"h": highs, "c": closes, "atr": atr})
    df = df.with_columns(
        patterns.breakout_reversal_up(
            pl.col("h"), pl.col("c"), pl.col("atr"),
            lookback_bars=30, reversal_atr=0.5,
        ).alias("bru")
    )
    assert df["bru"][-1] == 0


# ===========================================================================
# T7.07 Spike-and-fade volume
# ===========================================================================

def test_spike_and_fade_fires():
    # Baseline ~100, spike to 1000, then back to 10
    vols = [100] * 20 + [1000, 10, 100, 100]
    df = pl.DataFrame({"v": vols})
    df = df.with_columns(
        patterns.spike_and_fade_volume(pl.col("v"), lookback_bars=20, spike_multiplier=3.0).alias("sf")
    )
    # Spike bar at idx 20 → flag = 1 (current > 3×baseline AND next < baseline/3)
    assert df["sf"][20] == 1


# ===========================================================================
# T7.08 Imbalance persistence run-length
# ===========================================================================

def test_imbalance_persistence_all_positive():
    """All-positive imbalance over window → run-length = window."""
    df = pl.DataFrame({"imb": [0.5] * 30})
    df = df.with_columns(
        patterns.imbalance_persistence_runlength(pl.col("imb"), window=20).alias("ipl")
    )
    # Last row: last 20 all +1 → sum = 20
    assert df["ipl"][-1] == 20


def test_imbalance_persistence_mixed():
    df = pl.DataFrame({"imb": [0.5, 0.5, -0.5, -0.5, 0.5]})
    df = df.with_columns(
        patterns.imbalance_persistence_runlength(pl.col("imb"), window=5).alias("ipl")
    )
    # Last row sums sign over last 5: +1+1-1-1+1 = 1
    assert df["ipl"][-1] == 1


# ===========================================================================
# T7.09 CVD / price divergence
# ===========================================================================

def test_cvd_price_divergence_up_fires():
    """Price makes new 30-bar high, but CVD doesn't → divergence flag."""
    n = 35
    prices = [100.0] * 30 + [101.0, 102.0, 103.0, 104.0, 105.0]   # new highs starting bar 30
    cvds = [0.0] * 35  # CVD stays flat → definitely NOT new high
    df = pl.DataFrame({"p": prices, "cvd": cvds})
    df = df.with_columns(
        patterns.cvd_price_divergence_up(pl.col("cvd"), pl.col("p"), window=30).alias("div")
    )
    # At bar 30 (new price high, CVD not new high) → flag 1
    assert df["div"][30] == 1


# ===========================================================================
# T7.10 Range compression
# ===========================================================================

def test_range_compression_tight():
    """Tight HL bars → low compression ratio."""
    n = 30
    df = pl.DataFrame({
        "h": [100.1] * n,  # very tight range
        "l": [99.9] * n,
        "atr": [1.0] * n,
    })
    df = df.with_columns(
        patterns.range_compression_ratio(
            pl.col("h"), pl.col("l"), pl.col("atr"), window=10
        ).alias("rc")
    )
    # std of constant 0.2 range / 1.0 ATR = 0 (std of constants is 0)
    valid = df["rc"].drop_nulls().to_list()
    assert all(abs(v) < 1e-9 for v in valid)


# ===========================================================================
# T7.12 Hidden liquidity rolling ratio
# ===========================================================================

def test_hidden_liquidity_rolling_ratio():
    """Constant hidden/total → constant ratio."""
    n = 30
    df = pl.DataFrame({
        "hidden": [10.0] * n,
        "total": [100.0] * n,
    })
    df = df.with_columns(
        patterns.hidden_liquidity_rolling_ratio(
            pl.col("hidden"), pl.col("total"), window=10
        ).alias("hlr")
    )
    valid = df["hlr"].drop_nulls().to_list()
    # ratio = 100/1000 = 0.1
    assert all(abs(v - 0.1) < 1e-9 for v in valid)
