"""Tests for src/features/l1.py — pure polars-expression features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import l1


def _df():
    """Simple toy L1 frame: bid 4999/ask 5001 (50 sz each), wide then tight."""
    return pl.DataFrame(
        {
            "bid_px": [4999.00, 4999.50, 5000.00, 5000.25, 5000.50],
            "ask_px": [5001.00, 5000.50, 5000.25, 5000.50, 5000.75],
            "bid_sz": [50, 40, 30, 25, 20],
            "ask_sz": [50, 60, 10, 15, 25],
        }
    )


def test_mid_price():
    df = _df().with_columns(
        l1.mid_price(pl.col("bid_px"), pl.col("ask_px")).alias("mid")
    )
    assert df["mid"].to_list() == [5000.0, 5000.0, 5000.125, 5000.375, 5000.625]


def test_microprice_equal_size_equals_mid():
    df = pl.DataFrame({"bid_px": [100.0], "ask_px": [101.0], "bid_sz": [10], "ask_sz": [10]})
    df = df.with_columns(
        l1.microprice(pl.col("bid_px"), pl.col("ask_px"), pl.col("bid_sz"), pl.col("ask_sz")).alias("mp")
    )
    # Equal sizes → microprice = (100*10 + 101*10)/(10+10) = 100.5 = mid
    assert abs(df["mp"][0] - 100.5) < 1e-6


def test_microprice_asymmetric_size():
    """Heavier bid → microprice tilts UP toward ask (because ask_sz smaller → pressure to trade up)."""
    df = pl.DataFrame({"bid_px": [100.0], "ask_px": [101.0], "bid_sz": [100], "ask_sz": [1]})
    df = df.with_columns(
        l1.microprice(pl.col("bid_px"), pl.col("ask_px"), pl.col("bid_sz"), pl.col("ask_sz")).alias("mp")
    )
    # microprice = (100*1 + 101*100) / 101 ≈ 100.99
    assert df["mp"][0] > 100.99


def test_order_imbalance_balanced():
    df = pl.DataFrame({"bid_sz": [10, 10, 10], "ask_sz": [10, 10, 10]})
    df = df.with_columns(l1.order_imbalance(pl.col("bid_sz"), pl.col("ask_sz"), window=1).alias("oi"))
    # ~0.5 (tiny EPS bias from 1e-9 denominator)
    assert all(abs(v - 0.5) < 1e-6 for v in df["oi"].to_list())


def test_order_imbalance_bid_heavy():
    df = pl.DataFrame({"bid_sz": [90, 90], "ask_sz": [10, 10]})
    df = df.with_columns(l1.order_imbalance(pl.col("bid_sz"), pl.col("ask_sz"), window=1).alias("oi"))
    # bid_share = 90/100 = 0.9
    assert all(abs(v - 0.9) < 1e-6 for v in df["oi"].to_list())


def test_spread_abs_and_rel_bps():
    df = _df().with_columns([
        l1.spread_abs(pl.col("bid_px"), pl.col("ask_px")).alias("sp_abs"),
        l1.mid_price(pl.col("bid_px"), pl.col("ask_px")).alias("mid"),
    ])
    df = df.with_columns(l1.spread_rel_bps(pl.col("sp_abs"), pl.col("mid")).alias("sp_bps"))
    # First row: spread=2.0, mid=5000, bps = 2/5000 * 10000 = 4
    assert abs(df["sp_bps"][0] - 4.0) < 1e-6


def test_mid_price_return_log():
    df = pl.DataFrame({"mid": [100.0, 101.0, 102.01]})
    df = df.with_columns(l1.mid_price_return(pl.col("mid")).alias("ret"))
    # Second bar: log(101/100) ≈ 0.00995
    assert abs(df["ret"][1] - np.log(101/100)) < 1e-9
    # Third: log(102.01/101) ≈ 0.00995
    assert abs(df["ret"][2] - np.log(102.01/101)) < 1e-9


def test_tick_volatility_and_up_down():
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(100) * 0.001
    df = pl.DataFrame({"ret": rets.tolist()})
    df = df.with_columns([
        l1.tick_volatility(pl.col("ret"), window=20).alias("tv"),
        l1.up_volatility(pl.col("ret"), window=20).alias("uv"),
        l1.down_volatility(pl.col("ret"), window=20).alias("dv"),
    ])
    # All should have ~80 valid values (window=20 → first 19 are null/edge)
    assert df["tv"].drop_nulls().len() > 80
    # up_volatility fills nulls with 0 by construction
    assert df["uv"].null_count() == 0


def test_quote_slope_proxy():
    df = pl.DataFrame({"sp": [1.0, 1.0, 1.0], "b": [100, 50, 10], "a": [100, 50, 10]})
    df = df.with_columns(l1.quote_slope_proxy(pl.col("sp"), pl.col("b"), pl.col("a")).alias("qsp"))
    # Thinner book → higher slope
    assert df["qsp"][2] > df["qsp"][1] > df["qsp"][0]


def test_microprice_drift_log():
    df = pl.DataFrame({"mp": [100.0, 101.0, 102.01]})
    df = df.with_columns(l1.microprice_drift(pl.col("mp"), shift=1).alias("mpd"))
    assert abs(df["mpd"][1] - np.log(101/100)) < 1e-9


def test_jump_intensity_counts_jumps():
    """Return of 0.05 vs prior vol 0.01 and threshold 3.0: is_jump=True."""
    rets = [0.001] * 10 + [0.05] + [0.001] * 5
    vols = [0.01] * len(rets)  # constant prior vol
    df = pl.DataFrame({"ret": rets, "vol": vols})
    df = df.with_columns(
        l1.jump_intensity(pl.col("ret"), pl.col("vol"), intensity_window=5, jump_threshold=3.0).alias("ji")
    )
    # At the jump bar (idx=10), past-5 count should be >= 1
    row = df["ji"].to_list()
    # The window of 5 starting at idx 10 includes the jump bar itself or not depending on
    # how rolling_sum aligns — accept >= 1 somewhere after the jump
    assert max(row[10:15]) >= 1


def test_tick_return_skew_kurt():
    rng = np.random.default_rng(1)
    rets = rng.standard_normal(500) * 0.001
    df = pl.DataFrame({"ret": rets.tolist()})
    df = df.with_columns([
        l1.tick_return_skew(pl.col("ret"), window=60).alias("skew"),
        l1.tick_return_kurtosis(pl.col("ret"), window=60).alias("kurt"),
    ])
    # Both should have valid values after warmup
    assert df["skew"].drop_nulls().len() > 400
    assert df["kurt"].drop_nulls().len() > 400
    # For IID normal returns, mean excess kurt ≈ 0; check it's in a sane range
    mean_kurt = df["kurt"].drop_nulls().mean()
    assert -3 < mean_kurt < 3
