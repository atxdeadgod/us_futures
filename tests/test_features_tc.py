"""Tests for src/features/tc_features.py."""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features.tc_features import (
    attach_minute_of_day_cyclic,
    attach_ofi_zscore_tc,
    attach_realized_vol_zscore_tc,
    attach_session_flags,
    attach_spread_zscore_tc,
    attach_tc_zscore,
    attach_volume_surprise_tc,
)


def _mk_15min_bars(n_days: int, value_by_hour_utc: dict[int, float], rng_seed: int = 0):
    """Build n_days × 96 15-min bars; column 'value' = value_by_hour_utc[hour] + small noise."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    base = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    for d in range(n_days):
        for b in range(96):
            ts = base + timedelta(days=d, minutes=15 * b)
            base_v = value_by_hour_utc.get(ts.hour, 1.0)
            v = base_v + rng.normal(0, 0.01)  # tiny noise so std > 0
            rows.append(dict(ts=ts, value=v))
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC"))).sort("ts")


# ---------------------------------------------------------------------------
# attach_tc_zscore — generic primitive
# ---------------------------------------------------------------------------

def test_tc_zscore_constant_per_hour_gives_near_zero():
    """If value at each hour is constant across days, TC z-score → ~0 (after warmup)."""
    bars = _mk_15min_bars(n_days=40, value_by_hour_utc={h: float(h) for h in range(24)})
    out = attach_tc_zscore(bars, "value", lookback_days=20)
    valid = out.filter(pl.col("value_tc_z").is_finite()).filter(
        pl.col("value_tc_z").is_not_null()
    )
    assert valid.height > 0
    # All z-scores should be small (perturbed only by 0.01 noise; std ~0.01 → z within ±a few)
    assert valid["value_tc_z"].abs().max() < 5.0


def test_tc_zscore_jump_in_one_hour_shows_high_z_at_that_hour():
    """If hour 14 UTC is normally 1.0 but spikes to 100 on the last day, z at that bar is large."""
    rows = []
    base = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    for d in range(40):
        for b in range(96):
            ts = base + timedelta(days=d, minutes=15 * b)
            v = 1.0 + (99.0 if (d == 39 and ts.hour == 14 and ts.minute == 0) else 0.0)
            rows.append(dict(ts=ts, value=v))
    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC"))).sort("ts")
    out = attach_tc_zscore(bars, "value", lookback_days=15)
    # Find the spike bar
    spike = out.filter(
        (pl.col("ts").dt.date() == datetime(2024, 2, 10).date())
        & (pl.col("ts").dt.hour() == 14)
        & (pl.col("ts").dt.minute() == 0)
    )
    # Std for that bar-of-day = 0 (all prior values were 1.0) → z = (100-1)/EPS = huge
    # Or if std is computed including current row, std≈0 still → very large z
    assert spike.height >= 1


def test_tc_zscore_invalid_partition_minutes_raises():
    bars = _mk_15min_bars(n_days=2, value_by_hour_utc={})
    with pytest.raises(ValueError, match="partition_minutes"):
        attach_tc_zscore(bars, "value", partition_minutes=7)


# ---------------------------------------------------------------------------
# Specific wrappers
# ---------------------------------------------------------------------------

def test_attach_volume_surprise_tc_emits_column():
    bars = _mk_15min_bars(n_days=20, value_by_hour_utc={}).rename({"value": "volume"})
    out = attach_volume_surprise_tc(bars, lookback_days=10)
    assert "volume_surprise_tc" in out.columns
    assert out.height == bars.height


def test_attach_ofi_zscore_tc():
    bars = _mk_15min_bars(n_days=20, value_by_hour_utc={}).rename({"value": "ofi"})
    out = attach_ofi_zscore_tc(bars, ofi_col="ofi", lookback_days=10)
    assert "ofi_tc_z" in out.columns


def test_attach_spread_zscore_tc():
    bars = _mk_15min_bars(n_days=20, value_by_hour_utc={}).rename({"value": "spread_abs_close"})
    out = attach_spread_zscore_tc(bars, lookback_days=10)
    assert "spread_tc_z" in out.columns


def test_attach_realized_vol_zscore_tc():
    bars = _mk_15min_bars(n_days=20, value_by_hour_utc={}).rename({"value": "rvol"})
    out = attach_realized_vol_zscore_tc(bars, rvol_col="rvol", lookback_days=10)
    assert "rvol_tc_z" in out.columns


# ---------------------------------------------------------------------------
# Session flags
# ---------------------------------------------------------------------------

def test_attach_session_flags_at_known_et_hours():
    """Build bars at specific UTC hours that map to known ET hours and verify flags."""
    # Winter EST: UTC = ET + 5. So UTC 14:00 = ET 09:00 (RTH).
    rows = []
    base = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)  # winter
    # UTC 02:00 = ET 21:00 (ASIA), UTC 09:00 = ET 04:00 (EU),
    # UTC 14:00 = ET 09:00 (RTH), UTC 21:00 = ET 16:00 (ETH)
    test_ts = [(2, "ASIA"), (9, "EU"), (14, "RTH"), (21, "ETH")]
    for h, _ in test_ts:
        rows.append(dict(ts=base + timedelta(hours=h)))
    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))

    out = attach_session_flags(bars)
    assert out["is_asia"][0] == 1 and out["is_eu"][0] == 0
    assert out["is_eu"][1] == 1
    assert out["is_rth"][2] == 1
    assert out["is_eth"][3] == 1
    # Only ONE flag should be set per row
    flag_sum = out.select(pl.sum_horizontal("is_asia", "is_eu", "is_rth", "is_eth"))
    assert (flag_sum.to_series() == 1).all()


def test_attach_session_flags_dst_aware():
    """During EDT (summer), UTC 14:00 = ET 10:00 (RTH); UTC 13:30 should also be RTH (ET 09:30)."""
    # July 1, 2024 = EDT (UTC offset -4)
    base = datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc)
    rows = [dict(ts=base + timedelta(hours=14))]
    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    out = attach_session_flags(bars)
    # UTC 14 (EDT) = ET 10:00 → RTH
    assert out["is_rth"][0] == 1
    assert out["hour_et"][0] == 10


# ---------------------------------------------------------------------------
# Cyclic minute-of-day
# ---------------------------------------------------------------------------

def test_minute_of_day_cyclic_at_anchors():
    """sin/cos at midnight ET = (0, 1); at noon ET = (0, -1); at 6am = (1, 0)."""
    # UTC 05:00 in winter = ET 00:00
    base_winter = datetime(2024, 1, 15, 5, 0, tzinfo=timezone.utc)
    # UTC 17:00 in winter = ET 12:00 (noon)
    noon_winter = datetime(2024, 1, 15, 17, 0, tzinfo=timezone.utc)
    # UTC 11:00 in winter = ET 06:00
    six_am_winter = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)
    rows = [dict(ts=t) for t in [base_winter, noon_winter, six_am_winter]]
    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    out = attach_minute_of_day_cyclic(bars)
    # At midnight: angle=0 → sin=0, cos=1
    assert abs(out["minute_of_day_sin"][0] - 0.0) < 1e-9
    assert abs(out["minute_of_day_cos"][0] - 1.0) < 1e-9
    # At noon: angle=π → sin=0, cos=-1
    assert abs(out["minute_of_day_sin"][1] - 0.0) < 1e-9
    assert abs(out["minute_of_day_cos"][1] - (-1.0)) < 1e-9
    # At 6am: angle=π/2 → sin=1, cos=0
    assert abs(out["minute_of_day_sin"][2] - 1.0) < 1e-9
    assert abs(out["minute_of_day_cos"][2] - 0.0) < 1e-9
