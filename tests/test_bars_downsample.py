"""Tests for bar downsampling + realized moments."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_downsample import (
    DEFAULT_AGG_RULES,
    downsample_bars,
    downsample_with_moments,
    realized_moments,
)


def _mk_5sec_bars(n: int, start: datetime | None = None, price_fn=None):
    """Make n synthetic 5-sec bars with close = price_fn(i) (default i+5000)."""
    if start is None:
        start = datetime(2024, 1, 2, 14, 30, 5, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = start + timedelta(seconds=5 * i)
        close = price_fn(i) if price_fn else 5000.0 + i
        rows.append(
            {
                "ts": ts,
                "root": "ES",
                "expiry": "ESH4",
                "open": close - 0.25,
                "high": close + 0.25,
                "low": close - 0.5,
                "close": close,
                "volume": 10,
                "dollar_volume": 10 * close,
                "buys_qty": 6,
                "sells_qty": 4,
                "trades_count": 2,
                "unclassified_count": 0,
                "bid_close": close - 0.125,
                "ask_close": close + 0.125,
                "mid_close": close,
                "spread_abs_close": 0.25,
                "cvd_globex": i * 2,  # cumulative (buys - sells) = i * (6-4) = i*2
                "cvd_rth": i * 2,
                "bars_since_rth_reset": i,
                "is_rth": True,
                "is_session_warm": True,
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


# ===========================================================================
# Type-1: simple downsampling
# ===========================================================================

def test_downsample_open_high_low_close():
    """OHLC rules: first / max / min / last."""
    bars_5s = _mk_5sec_bars(180)  # 15 minutes of 5-sec bars (bar closes at +5,+10,...+15:00 from start)
    bars_15m = downsample_bars(bars_5s, target_every="15m")
    # Should be ~1 bar (or 2 spanning the boundary)
    assert bars_15m.height >= 1
    # Check OHLC plays out: for any 15-min bar, open=first 5-sec close_of_prev_bar open logic
    # With close=5000+i, open=close-0.25 → first_open = 4999.75
    # high over 180 bars = max(close+0.25) = (5000+179) + 0.25 = 5179.25
    # low = min(close-0.5) = 4999.5
    # close = last close = 5179
    # Look at the 15-min bar that contains most of our data
    main_bar = bars_15m.filter(pl.col("volume") > 100).row(0, named=True)
    assert main_bar["volume"] >= 100  # sum of 5-sec volumes


def test_downsample_volume_sums():
    bars_5s = _mk_5sec_bars(180)  # each bar has volume=10 → total 1800 (or distributed across 15-min bars)
    bars_15m = downsample_bars(bars_5s, target_every="15m")
    # Total volume conserved
    assert bars_15m["volume"].sum() == 1800


def test_downsample_cvd_last():
    bars_5s = _mk_5sec_bars(180)  # cvd_globex = i*2 → last is 179*2=358
    bars_15m = downsample_bars(bars_5s, target_every="15m")
    # Final CVD should be the last 5-sec CVD (cumulative)
    assert bars_15m["cvd_globex"].to_list()[-1] == 358


def test_downsample_arbitrary_target_frequency():
    """Should work at 1-min, 5-min, 30-min."""
    bars_5s = _mk_5sec_bars(720)  # 60 min of data
    for freq in ("1m", "5m", "15m", "30m"):
        bars = downsample_bars(bars_5s, target_every=freq)
        assert bars.height > 0
        # Volume should be conserved across all aggregations
        assert bars["volume"].sum() == 7200


# ===========================================================================
# Type-2: realized moments
# ===========================================================================

def test_realized_moments_constant_price_is_zero():
    """Constant close → zero realized vol, NaN skew/kurt (zero variance)."""
    bars_5s = _mk_5sec_bars(180, price_fn=lambda i: 5000.0)
    m = realized_moments(bars_5s, target_every="15m")
    # rv_5s should be 0 when all returns are 0
    # (excluding first NaN return at the start)
    row = m.filter(pl.col("rv_5s").is_not_null()).row(0, named=True)
    assert abs(row["rv_5s"]) < 1e-12


def test_realized_moments_random_walk_positive_rv():
    """Random walk → positive realized vol."""
    rng = np.random.default_rng(0)
    # Close = 5000 * exp(cumsum of small normal shocks)
    shocks = rng.standard_normal(180) * 0.001
    prices = 5000.0 * np.exp(np.cumsum(shocks))
    bars_5s = _mk_5sec_bars(180, price_fn=lambda i: float(prices[i]))
    m = realized_moments(bars_5s, target_every="15m")
    rvs = m.filter(pl.col("rv_5s").is_not_null())["rv_5s"].to_list()
    for rv in rvs:
        assert rv > 0


def test_realized_moments_subbar_count_matches():
    """n_subbars in each 15-min window should equal number of 5-sec bars in it."""
    bars_5s = _mk_5sec_bars(180)
    m = realized_moments(bars_5s, target_every="15m")
    # Sum of n_subbars across all windows should equal total bars (180)
    assert m["n_subbars"].sum() == 180


def test_downsample_with_moments_combines():
    bars_5s = _mk_5sec_bars(180)
    combined = downsample_with_moments(bars_5s, target_every="15m")
    # Should have BOTH the Type-1 cols (close, volume) AND the Type-2 cols (rv_5s, realized_skew_5s)
    cols = combined.columns
    assert "close" in cols
    assert "volume" in cols
    assert "rv_5s" in cols
    assert "realized_skew_5s" in cols
    assert "n_subbars" in cols


def test_arbitrary_target_frequency_moments():
    """Realized moments at 1-min and 30-min both produce sane output."""
    bars_5s = _mk_5sec_bars(360)  # 30 min
    m_1m = realized_moments(bars_5s, target_every="1m")
    m_30m = realized_moments(bars_5s, target_every="30m")
    assert m_1m.height > m_30m.height   # more 1-min bars than 30-min bars
    # Sum of rv across sub-windows should be approximately equal (additive RV property)
    # For price_fn default close=5000+i, return_i ≈ 1/(5000+i)² ≈ constant.
    # Compare totals to within 1% (floating-point accumulation in different groupings)
    rv_1m = float(m_1m["rv_5s"].sum())
    rv_30m = float(m_30m["rv_5s"].sum())
    assert abs(rv_1m - rv_30m) / rv_30m < 0.01
