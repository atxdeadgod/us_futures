"""Tests for src/features/bar.py."""
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

from src.features import bar


def _price_series(n=100, start=100.0, drift=0.001, seed=0):
    """Log-normal random-walk price series."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n) * 0.01 + drift
    prices = start * np.exp(np.cumsum(shocks))
    return prices


def test_log_return_horizon_one():
    df = pl.DataFrame({"close": [100.0, 101.0, 102.01]})
    df = df.with_columns(bar.log_return(pl.col("close"), horizon=1).alias("r"))
    assert abs(df["r"][1] - np.log(101/100)) < 1e-9


def test_log_return_horizon_2():
    df = pl.DataFrame({"close": [100.0, 101.0, 102.01]})
    df = df.with_columns(bar.log_return(pl.col("close"), horizon=2).alias("r"))
    assert abs(df["r"][2] - np.log(102.01/100)) < 1e-9


def test_log_volume_basic():
    df = pl.DataFrame({"vol": [0, 1, 100]})
    df = df.with_columns(bar.log_volume(pl.col("vol")).alias("lv"))
    assert abs(df["lv"][0] - np.log(1)) < 1e-9   # log(1+0)=0
    assert abs(df["lv"][1] - np.log(2)) < 1e-9


def test_realized_volatility_std():
    rets = [0.01, -0.01, 0.02, -0.02, 0.01]
    df = pl.DataFrame({"r": rets})
    df = df.with_columns(bar.realized_volatility(pl.col("r"), window=5, method="std").alias("rv"))
    # Last row has full window; std of [0.01,-0.01,0.02,-0.02,0.01]
    expected = float(np.std(rets))
    assert abs(df["rv"][4] - expected) < 1e-9


def test_realized_volatility_ewma():
    rets = [0.01] * 10
    df = pl.DataFrame({"r": rets})
    df = df.with_columns(bar.realized_volatility(pl.col("r"), window=5, method="ewma").alias("rv"))
    # Constant returns → ewma(r²) → r² → sqrt(r²) = |r| = 0.01
    assert abs(df["rv"][-1] - 0.01) < 1e-9


def test_parkinson_vs_gk():
    # Build OHLC with H > L > O = C
    n = 50
    opens = [100.0] * n
    closes = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    df = pl.DataFrame({"o": opens, "h": highs, "l": lows, "c": closes})
    df = df.with_columns([
        bar.range_vol_parkinson(pl.col("h"), pl.col("l"), window=10).alias("park"),
        bar.range_vol_gk(pl.col("o"), pl.col("h"), pl.col("l"), pl.col("c"), window=10).alias("gk"),
    ])
    # With H/L ratio = 101/99, vol should be positive for both after warmup
    assert df["park"][-1] > 0
    assert df["gk"][-1] > 0


def test_volatility_ratio_shortlong():
    """Short vol = 1.0, long vol = 2.0 → ratio = 0.5."""
    df = pl.DataFrame({"short": [1.0, 1.0, 1.0], "long": [2.0, 2.0, 2.0]})
    df = df.with_columns(bar.volatility_ratio(pl.col("short"), pl.col("long")).alias("vr"))
    assert all(abs(v - 0.5) < 1e-9 for v in df["vr"].to_list())


def test_jump_indicator_flag():
    """|ret| >> 3 × prev-vol → jump flag = 1."""
    rets = [0.001] * 5 + [0.10] + [0.001] * 5  # big jump at idx 5
    vols = [0.001] * len(rets)  # constant prior vol
    df = pl.DataFrame({"r": rets, "v": vols})
    df = df.with_columns(
        bar.jump_indicator(pl.col("r"), pl.col("v"), threshold=3.0, output="flag").alias("j")
    )
    assert df["j"][5] == 1  # jump detected
    assert df["j"][1] == 0  # normal bars no jump


def test_return_autocorrelation_sane():
    """IID returns → ~zero autocorrelation."""
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(500) * 0.001
    df = pl.DataFrame({"r": rets.tolist()})
    df = df.with_columns(
        bar.return_autocorrelation(pl.col("r"), lag=1, window=100).alias("ac")
    )
    mean_ac = df["ac"].drop_nulls().mean()
    assert abs(mean_ac) < 0.15  # small for IID


def test_vwap_return_nonzero():
    """Varying prices → non-zero VWAP return after warmup."""
    n = 50
    df = pl.DataFrame({
        "h": [100 + i * 0.1 for i in range(n)],
        "l": [99 + i * 0.1 for i in range(n)],
        "c": [99.5 + i * 0.1 for i in range(n)],
        "v": [100] * n,
    })
    df = df.with_columns(
        bar.vwap_return(pl.col("h"), pl.col("l"), pl.col("c"), pl.col("v"), window=10).alias("vr")
    )
    valid = df["vr"].drop_nulls()
    assert valid.len() > 0
    # Price is drifting up → most recent VWAP returns should be positive
    assert valid[-1] > 0


def test_volume_surprise_zscore():
    """Constant volume → zero surprise (eventually); spike → positive surprise."""
    vols = [100] * 10 + [500]  # spike at end
    df = pl.DataFrame({"lv": [np.log(1 + v) for v in vols]})
    df = df.with_columns(bar.volume_surprise(pl.col("lv"), window=10).alias("vs"))
    # Last row: big spike → positive z
    assert df["vs"][-1] > 2


def test_turnover_basic():
    """Rolling vol / shares."""
    df = pl.DataFrame({"v": [100] * 10, "so": [1000] * 10})
    df = df.with_columns(bar.turnover(pl.col("v"), pl.col("so"), window=5).alias("t"))
    # rolling_sum(5) = 500 / 1000 = 0.5
    assert abs(df["t"][-1] - 0.5) < 1e-6


def test_amihud_illiquidity_more_liquid_lower():
    """Bigger volume at same return → lower Amihud."""
    df_thin = pl.DataFrame({"r": [0.01] * 10, "v": [10] * 10, "c": [100.0] * 10})
    df_thick = pl.DataFrame({"r": [0.01] * 10, "v": [1000] * 10, "c": [100.0] * 10})
    df_thin = df_thin.with_columns(
        bar.amihud_illiquidity(pl.col("r"), pl.col("v"), pl.col("c"), window=5).alias("ai")
    )
    df_thick = df_thick.with_columns(
        bar.amihud_illiquidity(pl.col("r"), pl.col("v"), pl.col("c"), window=5).alias("ai")
    )
    assert df_thin["ai"][-1] > df_thick["ai"][-1]


def test_minute_of_day():
    """Handle i8 overflow correctly via Int32 cast."""
    ts_vals = [
        datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 23, 59, tzinfo=timezone.utc),
    ]
    df = pl.DataFrame({"ts": ts_vals}).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    df = df.with_columns(bar.minute_of_day(pl.col("ts")).alias("mod"))
    # 9:30 UTC = 570; 14:30 UTC = 870 (would overflow i8!); 23:59 UTC = 1439
    assert df["mod"].to_list() == [570, 870, 1439]


def test_is_monday_and_friday():
    # 2024-01-01 = Monday
    ts_vals = [datetime(2024, 1, d, tzinfo=timezone.utc) for d in (1, 2, 5, 8)]  # Mon, Tue, Fri, Mon
    df = pl.DataFrame({"ts": ts_vals}).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    df = df.with_columns([
        bar.is_monday(pl.col("ts")).alias("mon"),
        bar.is_friday(pl.col("ts")).alias("fri"),
    ])
    assert df["mon"].to_list() == [1, 0, 0, 1]
    assert df["fri"].to_list() == [0, 0, 1, 0]


def test_is_month_end():
    ts_vals = [datetime(2024, 1, d, tzinfo=timezone.utc) for d in (30, 31)]
    df = pl.DataFrame({"ts": ts_vals}).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    df = df.with_columns(bar.is_month_end(pl.col("ts")).alias("me"))
    assert df["me"].to_list() == [0, 1]


def test_vwap_deviation():
    df = pl.DataFrame({"close": [100.0, 101.0, 99.5], "vwap": [100.5, 100.5, 100.5]})
    df = df.with_columns(bar.vwap_deviation(pl.col("close"), pl.col("vwap")).alias("dev"))
    assert df["dev"].to_list() == [-0.5, 0.5, -1.0]
