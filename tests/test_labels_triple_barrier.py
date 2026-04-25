"""Tests for src/labels/triple_barrier.py."""
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

from src.labels.triple_barrier import (
    _balance_score,
    _triple_barrier_np,
    atr_column,
    attach_atr_time_conditional,
    triple_barrier_labels,
    tune_triple_barrier,
)


def _mk_bars(closes, highs=None, lows=None, opens=None):
    """Build a 5-sec bar frame from close/high/low/open arrays."""
    n = len(closes)
    highs = highs if highs is not None else [c + 0.5 for c in closes]
    lows = lows if lows is not None else [c - 0.5 for c in closes]
    opens = opens if opens is not None else [c for c in closes]
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(seconds=5 * i) for i in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


# ---------------------------------------------------------------------------
# atr_column
# ---------------------------------------------------------------------------

def test_atr_column_simple():
    """TR = H−L when no gap; rolling mean over window."""
    bars = _mk_bars([100.0] * 5, highs=[101.0] * 5, lows=[99.0] * 5)
    df = bars.with_columns(
        atr_column(pl.col("high"), pl.col("low"), pl.col("close"), window=3).alias("atr")
    )
    # Each bar TR=2.0; rolling_mean with window=3 → first 2 null, rest = 2.0
    vals = df["atr"].to_list()
    assert vals[0] is None and vals[1] is None
    assert all(abs(v - 2.0) < 1e-9 for v in vals[2:])


def test_atr_column_with_gap():
    """Gap up: TR should include |H - prev_close|."""
    closes = [100.0, 100.0, 110.0, 110.0]  # gap up at idx 2
    highs = [100.5, 100.5, 111.0, 110.5]
    lows = [99.5, 99.5, 109.5, 109.5]
    bars = _mk_bars(closes, highs=highs, lows=lows)
    df = bars.with_columns(
        atr_column(pl.col("high"), pl.col("low"), pl.col("close"), window=1).alias("atr")
    )
    # Idx 2: TR = max(111-109.5, |111-100|, |109.5-100|) = 11.0
    assert abs(df["atr"][2] - 11.0) < 1e-9


# ---------------------------------------------------------------------------
# _triple_barrier_np — direct numpy tests
# ---------------------------------------------------------------------------

def test_upper_barrier_hit_first():
    """Close rising smoothly → upper barrier hit first → label +1."""
    close = np.array([100.0, 100.5, 101.0, 102.0, 103.0, 104.0])
    high = np.array([100.5, 101.0, 101.5, 102.5, 103.5, 104.5])
    low = np.array([99.5, 100.0, 100.5, 101.5, 102.5, 103.5])
    open_ = close.copy()
    atr = np.array([1.0] * 6)
    labels, offsets, rets, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=1.5, k_dn=1.5, T=5
    )
    # From i=0: upper=101.5, lower=98.5. high[2]=101.5 hits upper first.
    assert labels[0] == 1
    assert offsets[0] == 2
    assert abs(rets[0] - np.log(101.5 / 100.0)) < 1e-9


def test_lower_barrier_hit_first():
    """Close falling → lower barrier hit first → label -1."""
    close = np.array([100.0, 99.5, 99.0, 98.0, 97.0, 96.0])
    high = np.array([100.5, 100.0, 99.5, 98.5, 97.5, 96.5])
    low = np.array([99.5, 99.0, 98.5, 97.5, 96.5, 95.5])
    open_ = close.copy()
    atr = np.array([1.0] * 6)
    labels, offsets, rets, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=1.5, k_dn=1.5, T=5
    )
    # From i=0: upper=101.5, lower=98.5. low[2]=98.5 hits lower first.
    assert labels[0] == -1
    assert offsets[0] == 2
    assert abs(rets[0] - np.log(98.5 / 100.0)) < 1e-9


def test_time_expired_zero_label():
    """Flat market within barriers → label 0 at vertical barrier T."""
    close = np.array([100.0] * 10)
    high = np.array([100.1] * 10)
    low = np.array([99.9] * 10)
    open_ = close.copy()
    atr = np.array([1.0] * 10)
    labels, offsets, rets, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=2.0, k_dn=2.0, T=4
    )
    # Barriers at ±2.0; market stays in ±0.1. Time-expired → label 0, offset=4
    assert labels[0] == 0
    assert offsets[0] == 4
    assert abs(rets[0] - 0.0) < 1e-9


def test_within_bar_ambiguity_up_close():
    """Both barriers hit in same bar, close > open → +1."""
    close = np.array([100.0, 102.0])
    high = np.array([100.5, 103.0])  # hits upper 101.5
    low = np.array([99.5, 98.0])  # hits lower 98.5
    open_ = np.array([100.0, 99.0])  # open < close at bar 1 → upward bar
    atr = np.array([1.0, 1.0])
    labels, offsets, rets, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=1.5, k_dn=1.5, T=3
    )
    # From i=0: both barriers hit at j=1. close[1]=102 > open[1]=99 → +1
    assert labels[0] == 1
    assert offsets[0] == 1


def test_within_bar_ambiguity_down_close():
    """Both barriers hit in same bar, close < open → -1."""
    close = np.array([100.0, 99.0])
    high = np.array([100.5, 103.0])  # hits upper 101.5
    low = np.array([99.5, 98.0])  # hits lower 98.5
    open_ = np.array([100.0, 102.0])  # open > close at bar 1 → downward bar
    atr = np.array([1.0, 1.0])
    labels, offsets, rets, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=1.5, k_dn=1.5, T=3
    )
    assert labels[0] == -1
    assert offsets[0] == 1


def test_nan_atr_gives_zero_label():
    """NaN ATR (warmup) → label 0, offset 0 — filtered downstream."""
    close = np.array([100.0, 101.0, 102.0])
    high = np.array([100.5, 101.5, 102.5])
    low = np.array([99.5, 100.5, 101.5])
    open_ = close.copy()
    atr = np.array([np.nan, 1.0, 1.0])
    labels, offsets, rets, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=1.5, k_dn=1.5, T=2
    )
    assert labels[0] == 0
    assert offsets[0] == 0
    assert np.isnan(rets[0])


# ---------------------------------------------------------------------------
# triple_barrier_labels — DataFrame wrapper
# ---------------------------------------------------------------------------

def test_triple_barrier_labels_schema():
    """Output DataFrame has expected columns & dtypes."""
    bars = _mk_bars([100.0 + 0.1 * i for i in range(40)])
    out = triple_barrier_labels(bars, k_up=1.0, k_dn=1.0, T=5, atr_window=5)
    for c in ("atr", "label", "hit_offset", "realized_ret", "realized_ret_pts", "halt_truncated"):
        assert c in out.columns
    assert out.schema["label"] == pl.Int8
    assert out.schema["hit_offset"] == pl.Int32
    assert out.schema["realized_ret"] == pl.Float64
    assert out.schema["realized_ret_pts"] == pl.Float64
    assert out.schema["halt_truncated"] == pl.Boolean
    assert out.height == bars.height


def test_realized_ret_pts_matches_barrier_distance():
    """For +1 label, realized_ret_pts = upper − close[i] = k_up * atr[i]."""
    close = np.array([100.0, 100.5, 101.0, 102.0, 103.0])
    high = np.array([100.5, 101.0, 101.5, 102.5, 103.5])
    low = np.array([99.5, 100.0, 100.5, 101.5, 102.5])
    open_ = close.copy()
    atr = np.array([1.0] * 5)
    labels, _, _, rets_pts, _ = _triple_barrier_np(
        close, high, low, open_, atr, k_up=1.5, k_dn=1.5, T=4
    )
    # i=0: +1 label; upper=101.5 → rets_pts = 101.5 - 100.0 = 1.5 = k_up*atr
    assert labels[0] == 1
    assert abs(rets_pts[0] - 1.5) < 1e-9


def test_triple_barrier_labels_upward_drift():
    """Steady upward drift → majority +1 labels."""
    n = 60
    closes = [100.0 + 0.5 * i for i in range(n)]  # drift up 0.5/bar
    highs = [c + 0.25 for c in closes]
    lows = [c - 0.25 for c in closes]
    bars = _mk_bars(closes, highs=highs, lows=lows)
    out = triple_barrier_labels(bars, k_up=1.0, k_dn=1.0, T=6, atr_window=5)
    valid = out.filter(pl.col("atr").is_not_null() & pl.col("realized_ret").is_not_null())
    frac_pos = (valid["label"] == 1).sum() / valid.height
    assert frac_pos > 0.7  # strong upward drift → most labels positive


# ---------------------------------------------------------------------------
# _balance_score
# ---------------------------------------------------------------------------

def test_balance_score_uniform_is_one():
    """Uniform 1/3 split → balance_score = 1."""
    s = _balance_score(1 / 3, 1 / 3, 1 / 3)
    assert abs(s - 1.0) < 1e-6


def test_balance_score_one_class_is_zero():
    """All one class → balance_score ≈ 0."""
    s = _balance_score(1.0, 0.0, 0.0)
    assert s < 1e-6


def test_balance_score_monotonic():
    """More uniform → higher score."""
    s_skewed = _balance_score(0.8, 0.1, 0.1)
    s_even = _balance_score(0.4, 0.3, 0.3)
    assert s_even > s_skewed


# ---------------------------------------------------------------------------
# tune_triple_barrier — grid search harness
# ---------------------------------------------------------------------------

def test_tune_triple_barrier_row_count():
    """Grid of 2×2×2×1 → 8 rows returned."""
    n = 80
    rng = np.random.default_rng(42)
    # Random walk with small drift
    rets = rng.normal(0.0, 0.3, n).cumsum()
    closes = (100.0 + rets).tolist()
    highs = [c + 0.3 for c in closes]
    lows = [c - 0.3 for c in closes]
    bars = _mk_bars(closes, highs=highs, lows=lows)
    out = tune_triple_barrier(
        bars,
        k_up_grid=(1.0, 2.0),
        k_dn_grid=(1.0, 2.0),
        T_grid=(4, 8),
        atr_window_grid=(10,),
    )
    assert out.height == 8
    expected_cols = {
        "k_up", "k_dn", "T", "atr_window",
        "frac_pos", "frac_neg", "frac_zero",
        "mean_ret_pos", "mean_ret_neg", "mean_ret_zero",
        "mean_ret_pts_pos", "mean_ret_pts_neg", "mean_ret_pts_zero",
        "mean_ret_bps_pos", "mean_ret_bps_neg", "mean_ret_bps_zero",
        "pts_over_cost_pos", "pts_over_cost_neg",
        "mean_hit_offset_pos", "mean_hit_offset_neg", "mean_hit_offset_zero",
        "n_total", "label_forward_return_corr", "balance_score",
    }
    assert expected_cols.issubset(set(out.columns))


def test_tune_triple_barrier_fractions_sum_to_one():
    """frac_pos + frac_neg + frac_zero = 1 for each row."""
    n = 60
    rng = np.random.default_rng(7)
    closes = (100.0 + rng.normal(0.0, 0.3, n).cumsum()).tolist()
    bars = _mk_bars(closes, highs=[c + 0.3 for c in closes], lows=[c - 0.3 for c in closes])
    out = tune_triple_barrier(
        bars, k_up_grid=(1.0,), k_dn_grid=(1.0,), T_grid=(4,), atr_window_grid=(10,)
    )
    row = out.row(0, named=True)
    assert abs(row["frac_pos"] + row["frac_neg"] + row["frac_zero"] - 1.0) < 1e-9


def test_tune_triple_barrier_corr_finite_with_warmup_nans():
    """Regression: warmup rows + tail rows leave realized_ret as NaN (not null).
    Filter must use is_finite() so NaN rows don't poison .mean() and corrcoef."""
    n = 80
    rng = np.random.default_rng(99)
    closes = (100.0 + rng.normal(0.0, 0.4, n).cumsum()).tolist()
    bars = _mk_bars(closes, highs=[c + 0.3 for c in closes], lows=[c - 0.3 for c in closes])
    out = tune_triple_barrier(
        bars, k_up_grid=(1.0,), k_dn_grid=(1.0,), T_grid=(4,), atr_window_grid=(20,),
    )
    row = out.row(0, named=True)
    # ATR window=20 → first 20 rows have NaN ATR → labels=0 ret=NaN → must be filtered.
    # Last ~T rows can have NaN ret (j_max == i+1) → must also be filtered.
    # If the filter is correct, corr is a real float (not NaN).
    assert not np.isnan(row["label_forward_return_corr"]), \
        "corr is NaN — likely is_not_null filter not catching NaN rows"
    # mean_ret_pts_zero must also be a real number (zero-class is largest sample)
    if row["frac_zero"] > 0:
        assert not np.isnan(row["mean_ret_pts_zero"]), \
            "mean_ret_pts_zero is NaN — NaN rows leaking into label=0 mean"


def test_tune_triple_barrier_pts_over_cost():
    """pts_over_cost = |mean_pts| / cost_pts. Pass cost=1.0, check scaling."""
    n = 120
    rng = np.random.default_rng(3)
    closes = (100.0 + rng.normal(0.0, 0.3, n).cumsum()).tolist()
    bars = _mk_bars(closes, highs=[c + 0.3 for c in closes], lows=[c - 0.3 for c in closes])
    out_c1 = tune_triple_barrier(
        bars, k_up_grid=(1.0,), k_dn_grid=(1.0,), T_grid=(8,), atr_window_grid=(10,),
        cost_pts=1.0,
    )
    out_c2 = tune_triple_barrier(
        bars, k_up_grid=(1.0,), k_dn_grid=(1.0,), T_grid=(8,), atr_window_grid=(10,),
        cost_pts=2.0,
    )
    r1 = out_c1.row(0, named=True)
    r2 = out_c2.row(0, named=True)
    # Halving cost → doubling ratio
    assert abs(r1["pts_over_cost_pos"] - 2 * r2["pts_over_cost_pos"]) < 1e-9
    # Ratio always non-negative
    assert r1["pts_over_cost_pos"] >= 0
    assert r1["pts_over_cost_neg"] >= 0


def test_tune_triple_barrier_larger_T_higher_label_density():
    """Larger vertical barrier T → more labels terminate via barrier (not zero)."""
    n = 120
    rng = np.random.default_rng(13)
    closes = (100.0 + rng.normal(0.0, 0.4, n).cumsum()).tolist()
    bars = _mk_bars(closes, highs=[c + 0.3 for c in closes], lows=[c - 0.3 for c in closes])
    out = tune_triple_barrier(
        bars, k_up_grid=(1.5,), k_dn_grid=(1.5,), T_grid=(3, 20), atr_window_grid=(10,)
    )
    by_T = {int(r["T"]): r for r in out.iter_rows(named=True)}
    # Longer horizon → fewer time-expired (zero) labels
    assert by_T[20]["frac_zero"] <= by_T[3]["frac_zero"]


# ---------------------------------------------------------------------------
# Time-conditional ATR
# ---------------------------------------------------------------------------

def _mk_15min_bars_multi_day(n_days: int, vol_by_hour: dict[int, float], rng_seed: int = 0):
    """Build 15-min bar series across n_days, with hour-of-day-keyed vol scale."""
    rng = np.random.default_rng(rng_seed)
    bars_per_day = 96  # 24h × 4
    rows = []
    base = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    px = 5000.0
    for d in range(n_days):
        for b in range(bars_per_day):
            ts = base + timedelta(days=d, minutes=15 * b)
            hour_utc = ts.hour
            vol = vol_by_hour.get(hour_utc, 1.0)
            move = rng.normal(0, vol)
            px = px + move
            high = px + abs(rng.normal(0, vol * 0.5))
            low = px - abs(rng.normal(0, vol * 0.5))
            rows.append(dict(ts=ts, open=px, high=high, low=low, close=px))
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


def test_atr_time_conditional_partitions_by_bar_of_day():
    """TC-ATR over a known vol-pattern recovers the per-hour vol scale."""
    # 60 days, vol=10.0 at UTC hour 14-16 (= ET 09-11 in winter EST),
    # vol=1.0 elsewhere. TC-ATR for those hours should be MUCH bigger than off-hours.
    high_vol_hours = {14: 10.0, 15: 10.0, 16: 10.0}
    bars = _mk_15min_bars_multi_day(n_days=60, vol_by_hour=high_vol_hours)

    out = attach_atr_time_conditional(
        bars, lookback_days=20, bar_minutes=15, out_col="atr_tc"
    )
    out = out.with_columns(
        atr_column(pl.col("high"), pl.col("low"), pl.col("close"), window=20).alias("atr_cal"),
        pl.col("ts").dt.hour().alias("h_utc"),
    ).filter(pl.col("atr_tc").is_not_null() & pl.col("atr_cal").is_not_null())

    by_hour = out.group_by("h_utc").agg(pl.col("atr_tc").mean()).sort("h_utc")
    h_to_tc = {int(r["h_utc"]): r["atr_tc"] for r in by_hour.iter_rows(named=True)}

    # TC-ATR at high-vol hour (14) should be MUCH bigger than at low-vol hour (4)
    assert h_to_tc[14] > h_to_tc[4] * 5, \
        f"TC-ATR should track per-hour vol regime; got h14={h_to_tc[14]:.2f}, h4={h_to_tc[4]:.2f}"


def test_triple_barrier_labels_atr_mode_time_conditional():
    """`atr_mode='time_conditional'` produces labels using TC-ATR."""
    bars = _mk_15min_bars_multi_day(n_days=40, vol_by_hour={14: 5.0, 15: 5.0})
    out = triple_barrier_labels(
        bars, k_up=1.0, k_dn=1.0, T=4,
        atr_mode="time_conditional", lookback_days=15,
    )
    assert "atr" in out.columns
    # Should have non-null ATR for bars deep enough into the series
    valid = out.filter(pl.col("atr").is_finite() & pl.col("realized_ret").is_finite())
    assert valid.height > 100


def test_triple_barrier_labels_invalid_atr_mode():
    bars = _mk_bars([100.0 + 0.1 * i for i in range(40)])
    with pytest.raises(ValueError, match="atr_mode"):
        triple_barrier_labels(bars, atr_mode="bogus")


# ---------------------------------------------------------------------------
# Halt-aware labeling
# ---------------------------------------------------------------------------

def test_triple_barrier_labels_halt_aware_drops_halt_crossing_bars():
    """Bars whose forward T-window crosses a halt (>30min ts gap) get NaN ret."""
    # 10 bars at 15-min cadence, then a 60-min gap (simulated halt), then 5 more bars
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    ts = []
    for i in range(10):
        ts.append(start + timedelta(minutes=15 * i))
    # 60-min gap = halt
    halt_end = ts[-1] + timedelta(minutes=60)
    for i in range(5):
        ts.append(halt_end + timedelta(minutes=15 * i))
    n = len(ts)
    closes = [100.0 + 0.05 * i for i in range(n)]
    bars = pl.DataFrame({
        "ts": ts,
        "open": closes,
        "high": [c + 0.3 for c in closes],
        "low": [c - 0.3 for c in closes],
        "close": closes,
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))

    # T=4 → bars at indices 6, 7, 8, 9 have forward windows that cross the halt
    out = triple_barrier_labels(
        bars, k_up=1.5, k_dn=1.5, T=4, atr_window=3, halt_aware=True,
    )
    # Bars whose forward window crosses halt → realized_ret NaN, hit_offset = -1
    halt_dropped = out.filter(pl.col("hit_offset") == -1)
    # Should be at least the bars at indices 6, 7, 8, 9 (forward 4 bars crosses halt)
    assert halt_dropped.height >= 1
    # All of these have NaN ret
    assert halt_dropped["realized_ret"].is_nan().all() or halt_dropped["realized_ret"].is_null().all()


def test_triple_barrier_labels_halt_aware_off_keeps_halt_crossing_bars():
    """halt_aware=False reverts to legacy behavior (no halt detection)."""
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=15 * i) for i in range(10)]
    halt_end = ts[-1] + timedelta(minutes=60)
    ts += [halt_end + timedelta(minutes=15 * i) for i in range(5)]
    n = len(ts)
    closes = [100.0] * n
    bars = pl.DataFrame({
        "ts": ts, "open": closes,
        "high": [c + 0.3 for c in closes], "low": [c - 0.3 for c in closes], "close": closes,
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))

    out = triple_barrier_labels(bars, k_up=1.5, k_dn=1.5, T=4, atr_window=3, halt_aware=False)
    # No bars should be marked halt-dropped
    assert (out["hit_offset"] == -1).sum() == 0


def _mk_pre_halt_bars(pre_halt_n: int, post_halt_n: int):
    """15-min bar series with `pre_halt_n` bars, a 60-min halt, then `post_halt_n` bars.

    Bars rise smoothly so a tight upper barrier is hittable before halt.
    """
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=15 * i) for i in range(pre_halt_n)]
    halt_end = ts[-1] + timedelta(minutes=60)
    ts += [halt_end + timedelta(minutes=15 * i) for i in range(post_halt_n)]
    n = len(ts)
    closes = [100.0 + 0.1 * i for i in range(n)]
    return pl.DataFrame({
        "ts": ts,
        "open": closes,
        "high": [c + 0.3 for c in closes],
        "low": [c - 0.3 for c in closes],
        "close": closes,
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


def test_triple_barrier_labels_halt_truncate_keeps_short_horizon_labels():
    """halt_mode='truncate': bar 1 step before halt with T=4 gets effective T=1
    (below default min_effective_T=2 → dropped); but bar earlier with effective
    T >= min_effective_T is kept and marked halt_truncated=True."""
    bars = _mk_pre_halt_bars(pre_halt_n=10, post_halt_n=5)
    out = triple_barrier_labels(
        bars, k_up=1.5, k_dn=1.5, T=4, atr_window=3,
        halt_aware=True, halt_mode="truncate", min_effective_T=2,
    )
    # The bar at index 6 has forward calendar window 7,8,9, then 10=post-halt.
    # j_halt=10, effective_T = 10-1 - 6 = 3 (>=2, kept truncated).
    # The bar at index 9 (last pre-halt) has forward window 10..13 all post-halt;
    # j_halt=10, effective_T = 10-1 - 9 = 0 (<2, dropped).
    # The bar at index 8: j_halt=10, effective_T = 1 (<2, dropped).
    assert out["hit_offset"][9] == -1, "last pre-halt bar should be dropped (effective T = 0)"
    assert out["hit_offset"][8] == -1, "second-to-last pre-halt bar dropped (effective T = 1)"
    # Bar 6 with effective T = 3 should be kept and marked truncated
    assert out["halt_truncated"][6], "bar 6 should be truncated (effective T = 3, >= min)"
    assert out["hit_offset"][6] != -1


def test_triple_barrier_labels_halt_truncate_no_halt_no_truncation():
    """When forward window doesn't cross a halt, halt_truncated=False everywhere."""
    bars = _mk_pre_halt_bars(pre_halt_n=20, post_halt_n=0)
    out = triple_barrier_labels(
        bars, k_up=1.5, k_dn=1.5, T=4, atr_window=3,
        halt_aware=True, halt_mode="truncate", min_effective_T=2,
    )
    assert not out["halt_truncated"].any(), \
        "no bars should be marked halt_truncated when no halt is in forward window"


def test_triple_barrier_labels_invalid_halt_mode():
    bars = _mk_bars([100.0 + 0.1 * i for i in range(20)])
    with pytest.raises(ValueError, match="halt_mode"):
        triple_barrier_labels(bars, halt_mode="bogus")
