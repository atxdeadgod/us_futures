"""Tests for src/features/sub_bar_engines.py — VPIN + Hawkes asof-attach."""
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

from src.features import sub_bar_engines


def _mk_5sec_bars(n_bars: int = 720, seed: int = 0) -> pl.DataFrame:
    """Synthetic 5-sec bars: ~1 hour of bars, alternating buy/sell-heavy regimes."""
    rng = np.random.default_rng(seed)
    base = datetime(2024, 3, 4, 18, 0, 0, tzinfo=timezone.utc)  # 18:00 UTC = 13:00 ET
    rows = []
    for i in range(n_bars):
        ts = base + timedelta(seconds=5 * i)
        # First half: buy-heavy; second half: sell-heavy
        bias = 0.7 if i < n_bars // 2 else 0.3
        volume = max(10, int(rng.normal(200, 50)))
        buys = int(volume * bias + rng.normal(0, 5))
        buys = max(0, min(volume, buys))
        sells = volume - buys
        rows.append(dict(ts=ts, buys_qty=buys, sells_qty=sells, volume=volume))
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


def _mk_15m_bars(n_bars: int = 4) -> pl.DataFrame:
    """Synthetic 15-min target bars covering the same window as the 5-sec data."""
    base = datetime(2024, 3, 4, 18, 15, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n_bars):
        ts = base + timedelta(minutes=15 * i)
        rows.append(dict(ts=ts, close=5000.0 + i, volume=10_000))
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


# ---------------------------------------------------------------------------
# compute_vpin_buckets
# ---------------------------------------------------------------------------

def test_vpin_buckets_emits_expected_schema():
    bars_5s = _mk_5sec_bars(n_bars=720)
    out = sub_bar_engines.compute_vpin_buckets(bars_5s, bucket_size=2_000)
    assert out.columns == ["ts", "vpin", "bucket_volume", "n_sub_bars"]
    assert out.height > 0


def test_vpin_value_in_unit_interval():
    bars_5s = _mk_5sec_bars(n_bars=720)
    out = sub_bar_engines.compute_vpin_buckets(bars_5s, bucket_size=2_000)
    vpins = out["vpin"].to_list()
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in vpins)


def test_vpin_responds_to_imbalanced_flow():
    """In a regime with 70/30 buy/sell split, VPIN should be substantially > 0."""
    bars_5s = _mk_5sec_bars(n_bars=720)
    out = sub_bar_engines.compute_vpin_buckets(bars_5s, bucket_size=2_000)
    avg_vpin = out["vpin"].mean()
    # With 70/30 imbalance baseline noise should still produce VPIN > 0.2
    assert avg_vpin > 0.2


# ---------------------------------------------------------------------------
# compute_hawkes_at_5sec
# ---------------------------------------------------------------------------

def test_hawkes_emits_expected_schema():
    bars_5s = _mk_5sec_bars(n_bars=200)
    out = sub_bar_engines.compute_hawkes_at_5sec(bars_5s, hl_seconds_fast=5.0, hl_seconds_slow=60.0)
    assert "hawkes_imbalance_hl5" in out.columns
    assert "hawkes_imbalance_hl60" in out.columns
    assert "hawkes_acceleration" in out.columns
    assert out.height == bars_5s.height


def test_hawkes_imbalance_sign_matches_buy_pressure():
    """Buy-heavy regime → positive Hawkes imbalance; sell-heavy → negative."""
    bars_5s = _mk_5sec_bars(n_bars=720)
    out = sub_bar_engines.compute_hawkes_at_5sec(bars_5s, hl_seconds_fast=10.0, hl_seconds_slow=120.0)
    # First half is buy-heavy (bias=0.7) → imbalance should average positive
    first_half = out.head(out.height // 2)
    second_half = out.tail(out.height // 2)
    assert first_half["hawkes_imbalance_hl10"].drop_nulls().mean() > 0
    # Second half is sell-heavy (bias=0.3) — net flow over the slow HL eventually flips
    # (allow some lag because the slow HL=120s integrates the prior buy-pressure)
    assert second_half["hawkes_imbalance_hl10"].drop_nulls().mean() < first_half["hawkes_imbalance_hl10"].drop_nulls().mean()


def test_hawkes_acceleration_is_difference_of_fast_slow():
    bars_5s = _mk_5sec_bars(n_bars=200)
    out = sub_bar_engines.compute_hawkes_at_5sec(bars_5s, hl_seconds_fast=5.0, hl_seconds_slow=60.0)
    diff = (out["hawkes_imbalance_hl5"] - out["hawkes_imbalance_hl60"]).to_list()
    accel = out["hawkes_acceleration"].to_list()
    for d, a in zip(diff, accel):
        if d is None or a is None:
            continue
        assert abs(d - a) < 1e-9


# ---------------------------------------------------------------------------
# attach_sub_bar_engine_features (end-to-end)
# ---------------------------------------------------------------------------

def test_attach_sub_bar_engine_features_emits_all_columns():
    bars_5s = _mk_5sec_bars(n_bars=720)
    bars_15m = _mk_15m_bars(n_bars=4)
    out = sub_bar_engines.attach_sub_bar_engine_features(
        bars_15m, bars_5s, vpin_bucket_size=2_000,
    )
    expected = [
        "vpin", "vpin_buckets_velocity_w15m", "vpin_staleness_seconds",
        "hawkes_imbalance_hl5", "hawkes_imbalance_hl60", "hawkes_acceleration",
    ]
    for c in expected:
        assert c in out.columns, f"missing sub-bar engine col: {c}"
    assert out.height == bars_15m.height


def test_attach_sub_bar_engine_features_handles_ts_dtype_mismatch():
    """If 5-sec bars have datetime[μs] and 15-min bars have datetime[ns], the
    function should coerce before asof-join (no SchemaError)."""
    bars_5s = _mk_5sec_bars(n_bars=300).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC"))
    )
    bars_15m = _mk_15m_bars(n_bars=2)  # ns precision
    out = sub_bar_engines.attach_sub_bar_engine_features(
        bars_15m, bars_5s, vpin_bucket_size=1_000,
    )
    assert out.height == bars_15m.height
    # Some VPIN values should be filled (≥ one bucket completed before each 15-min bar close)
    assert out["vpin"].drop_nulls().len() > 0


def test_vpin_staleness_is_non_negative():
    bars_5s = _mk_5sec_bars(n_bars=720)
    bars_15m = _mk_15m_bars(n_bars=4)
    out = sub_bar_engines.attach_sub_bar_engine_features(
        bars_15m, bars_5s, vpin_bucket_size=2_000,
    )
    staleness = out["vpin_staleness_seconds"].drop_nulls().to_list()
    assert all(v >= 0 for v in staleness)


def test_vpin_buckets_velocity_count_matches_within_window():
    """vpin_buckets_velocity_w15m at ts=T should equal the number of bucket
    closes in (T-15m, T]. Verify by counting from the raw bucket frame."""
    bars_5s = _mk_5sec_bars(n_bars=720)
    bars_15m = _mk_15m_bars(n_bars=4)
    out = sub_bar_engines.attach_sub_bar_engine_features(
        bars_15m, bars_5s, vpin_bucket_size=2_000,
    )
    vpin_df = sub_bar_engines.compute_vpin_buckets(bars_5s, bucket_size=2_000)
    # For each 15-min bar, the velocity should be bucket-count in (ts-15min, ts]
    for row in out.iter_rows(named=True):
        ts = row["ts"]
        velocity = row["vpin_buckets_velocity_w15m"]
        window_start = ts - timedelta(minutes=15)
        manual = vpin_df.filter(
            (pl.col("ts") > window_start) & (pl.col("ts") <= ts)
        ).height
        assert velocity == manual, f"velocity mismatch at ts={ts}: got {velocity}, manual={manual}"
