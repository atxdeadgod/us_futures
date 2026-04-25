"""Tests for src/features/overnight.py."""
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

from src.features.overnight import attach_overnight_features
from src.features.tc_features import attach_session_flags


def _mk_bars_two_days(open_t1, close_t1_close, open_t2, prices_overnight, vols_overnight):
    """Build a 2-day bar series:
    - Day 1: 4 RTH bars (UTC 14:00-14:45 = ET 09:00-09:45) closing at close_t1_close
    - Overnight: bars at UTC 21:00, 22:00, 23:00, 00:00 (T+1) with given prices/vols
    - Day 2: 4 RTH bars opening at open_t2
    Returns a bar frame with `is_rth` already attached.
    """
    rows = []
    # Day 1 RTH (UTC 14:00-14:45 winter = ET 09:00-09:45)
    base1 = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
    px = open_t1
    rth_t1_prices = np.linspace(open_t1, close_t1_close, 4).tolist()
    for i, p in enumerate(rth_t1_prices):
        rows.append(dict(
            ts=base1 + timedelta(minutes=15*i),
            open=p, high=p+0.1, low=p-0.1, close=p, volume=100,
        ))

    # Overnight (4 bars at UTC 21:00, 22:00, 23:00, 00:00 = ET 16:00, 17:00, 18:00, 19:00)
    overnight_starts = [
        datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 15, 22, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 15, 23, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc),  # midnight UTC = 19:00 ET prev night
    ]
    for ts, p, v in zip(overnight_starts, prices_overnight, vols_overnight):
        rows.append(dict(ts=ts, open=p, high=p+0.1, low=p-0.1, close=p, volume=v))

    # Day 2 RTH (UTC 14:00 next day = ET 09:00)
    base2 = datetime(2024, 1, 16, 14, 0, tzinfo=timezone.utc)
    rth_t2_prices = np.linspace(open_t2, open_t2 + 1.0, 4).tolist()
    for i, p in enumerate(rth_t2_prices):
        rows.append(dict(
            ts=base2 + timedelta(minutes=15*i),
            open=p, high=p+0.1, low=p-0.1, close=p, volume=100,
        ))

    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC"))).sort("ts")
    return attach_session_flags(bars)


def test_overnight_log_return_basic():
    """yesterday close = 100, today open = 101 → overnight_log_return = log(101/100)."""
    bars = _mk_bars_two_days(
        open_t1=99, close_t1_close=100,
        open_t2=101,
        prices_overnight=[100, 100, 100, 100], vols_overnight=[10, 10, 10, 10],
    )
    out = attach_overnight_features(bars)
    # Find day-2 RTH bars (et_date = 2024-01-16)
    day2 = out.filter(pl.col("ts") >= datetime(2024, 1, 16, 14, 0, tzinfo=timezone.utc))
    expected = float(np.log(101.0 / 100.0))
    assert abs(day2["overnight_log_return"][0] - expected) < 1e-9
    # Constant within day 2
    assert day2["overnight_log_return"].n_unique() == 1


def test_overnight_realized_vol_sums_squared_returns():
    """Sum of squared 15-min log returns during overnight matches realized_vol²."""
    overnight_prices = [100, 102, 100, 103]
    bars = _mk_bars_two_days(
        open_t1=99, close_t1_close=100,
        open_t2=104,
        prices_overnight=overnight_prices, vols_overnight=[10, 10, 10, 10],
    )
    out = attach_overnight_features(bars)
    day2 = out.filter(pl.col("ts") >= datetime(2024, 1, 16, 14, 0, tzinfo=timezone.utc))
    rvol = float(day2["overnight_realized_vol"][0])
    assert rvol > 0


def test_overnight_volume_total():
    """Sum of overnight volumes is recorded correctly."""
    bars = _mk_bars_two_days(
        open_t1=99, close_t1_close=100, open_t2=101,
        prices_overnight=[100, 100, 100, 100], vols_overnight=[5, 7, 11, 13],
    )
    out = attach_overnight_features(bars)
    day2 = out.filter(pl.col("ts") >= datetime(2024, 1, 16, 14, 0, tzinfo=timezone.utc))
    assert day2["overnight_volume_total"][0] == 5 + 7 + 11 + 13


def test_overnight_n_bars_count():
    bars = _mk_bars_two_days(
        open_t1=99, close_t1_close=100, open_t2=101,
        prices_overnight=[100, 100, 100, 100], vols_overnight=[5, 5, 5, 5],
    )
    out = attach_overnight_features(bars)
    day2 = out.filter(pl.col("ts") >= datetime(2024, 1, 16, 14, 0, tzinfo=timezone.utc))
    assert day2["overnight_n_bars"][0] == 4


def test_overnight_features_first_day_has_null_return():
    """First trading date has no 'yesterday' so overnight_log_return is null."""
    bars = _mk_bars_two_days(
        open_t1=99, close_t1_close=100, open_t2=101,
        prices_overnight=[100, 100, 100, 100], vols_overnight=[5, 5, 5, 5],
    )
    out = attach_overnight_features(bars)
    day1 = out.filter(pl.col("ts") < datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc))
    # Day 1 has no prior overnight, so overnight_log_return should be null
    assert day1["overnight_log_return"][0] is None


def test_attach_overnight_requires_is_rth():
    """Must call attach_session_flags first."""
    rows = [dict(ts=datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc),
                 open=100.0, high=100.5, low=99.5, close=100.0, volume=100)]
    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    with pytest.raises(ValueError, match="is_rth"):
        attach_overnight_features(bars)
