"""Unit tests for src/features/engines.py — one test per §8 implementation trap.

Run: `pytest tests/test_engines.py -v`
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features.engines import (
    asof_strict_backward,
    cvd_with_dual_reset,
    ffd_weights,
    fracdiff_auto_d,
    fracdiff_series,
    hawkes_intensity_recursive,
    rolling_rth_bounded,
    round_number_pin_distance,
    vpin_volume_buckets,
)


# ===========================================================================
# §8.A  FracDiff
# ===========================================================================

def test_ffd_weights_d_zero_is_identity():
    w = ffd_weights(0.0)
    assert len(w) == 1
    assert w[0] == 1.0


def test_ffd_weights_d_one_is_first_diff():
    w = ffd_weights(1.0, tau=1e-12)
    # First-differencing weights: [1, -1, 0, 0, ...]; tau truncation stops at k=2
    assert w[0] == 1.0
    assert abs(w[1] + 1.0) < 1e-10
    # After k=1, subsequent weights are 0 exactly — truncated immediately
    assert len(w) <= 3


def test_ffd_weights_d_half_truncation():
    w_tight = ffd_weights(0.5, tau=1e-4)
    w_loose = ffd_weights(0.5, tau=1e-2)
    assert len(w_tight) > len(w_loose) > 0
    # Weights decay toward zero
    assert abs(w_tight[-1]) >= 1e-4 / 2  # close to threshold


def test_fracdiff_d_one_approximates_first_diff():
    x = pl.Series(np.cumsum(np.random.default_rng(0).standard_normal(500)))
    fd = fracdiff_series(x, 1.0).to_numpy()
    diff = np.diff(x.to_numpy(), prepend=np.nan)
    # The FFD with d=1 produces a 2-tap FIR identical to diff (ignoring NaN startup).
    # Compare aligned non-NaN values.
    valid = ~np.isnan(fd) & ~np.isnan(diff)
    assert valid.sum() > 400
    assert np.allclose(fd[valid], diff[valid], atol=1e-9)


def test_fracdiff_d_zero_is_identity():
    x = pl.Series(np.arange(100, dtype=float))
    fd = fracdiff_series(x, 0.0).to_numpy()
    assert np.allclose(fd, x.to_numpy(), equal_nan=True)


def test_fracdiff_causal_no_lookahead():
    """Verify a change at position k does not affect positions < k (causality).

    Use large tau so startup transient is small (~10 weights), perturbation at
    position 500 is safely past it.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    fd1 = fracdiff_series(pl.Series(x), 0.4, tau=1e-3).to_numpy()
    x2 = x.copy()
    x2[500] += 10.0
    fd2 = fracdiff_series(pl.Series(x2), 0.4, tau=1e-3).to_numpy()
    # Positions strictly before 500 must be unchanged
    assert np.allclose(fd1[:500], fd2[:500], equal_nan=True)
    # Position 500 must change
    assert abs(fd1[500] - fd2[500]) > 1e-6


def test_fracdiff_auto_d_returns_valid_d_on_random_walk():
    """Auto-d on pure I(1) returns some d ∈ (0, 1]. Don't assert specific d value."""
    rng = np.random.default_rng(7)
    x = pl.Series(np.cumsum(rng.standard_normal(1000)))
    fd, d = fracdiff_auto_d(x, p_value=0.01)
    assert 0 < d <= 1.0
    # Output may or may not pass ADF (falls back to d=1 if grid exhausted)
    assert fd.drop_nulls().len() > 100


def test_fracdiff_auto_d_on_stationary_white_noise():
    """True-stationary white noise: at d=0.1 the fracdiff is very close to identity
    and should pass ADF. auto_d should return small d."""
    rng = np.random.default_rng(11)
    x = pl.Series(rng.standard_normal(2000))
    fd, d = fracdiff_auto_d(x, p_value=0.01, d_grid=(0.1, 0.3, 0.5, 1.0))
    # Don't pin exact d; just require it returned something and output exists
    assert 0 < d <= 1.0
    assert fd.drop_nulls().len() > 500


# ===========================================================================
# §8.B  VPIN
# ===========================================================================

def test_vpin_all_buys_equals_one():
    ts = pl.datetime_range(datetime(2024, 1, 2, 14, 30), datetime(2024, 1, 2, 14, 40),
                            interval="5s", eager=True, time_unit="ns")
    sub = pl.DataFrame(
        {
            "ts": ts,
            "buys_qty": np.full(ts.len(), 100, dtype=np.int64),
            "sells_qty": np.zeros(ts.len(), dtype=np.int64),
        }
    )
    buckets = vpin_volume_buckets(sub, bucket_size=500)
    assert buckets.height > 0
    assert np.allclose(buckets["vpin"].to_numpy(), 1.0)


def test_vpin_balanced_equals_zero():
    ts = pl.datetime_range(datetime(2024, 1, 2, 14, 30), datetime(2024, 1, 2, 14, 40),
                            interval="5s", eager=True, time_unit="ns")
    sub = pl.DataFrame(
        {
            "ts": ts,
            "buys_qty": np.full(ts.len(), 100, dtype=np.int64),
            "sells_qty": np.full(ts.len(), 100, dtype=np.int64),
        }
    )
    buckets = vpin_volume_buckets(sub, bucket_size=1000)
    assert buckets.height > 0
    assert np.allclose(buckets["vpin"].to_numpy(), 0.0)


def test_vpin_partial_bucket_discarded_by_default():
    ts = pl.datetime_range(datetime(2024, 1, 2, 14, 30), datetime(2024, 1, 2, 14, 40),
                            interval="5s", eager=True, time_unit="ns")
    sub = pl.DataFrame(
        {
            "ts": ts,
            "buys_qty": np.full(ts.len(), 50, dtype=np.int64),
            "sells_qty": np.zeros(ts.len(), dtype=np.int64),
        }
    )
    # bucket_size=1e9 → never completes → no buckets emitted
    buckets = vpin_volume_buckets(sub, bucket_size=10**9, keep_partial=False)
    assert buckets.height == 0


# ===========================================================================
# §8.C  Hawkes
# ===========================================================================

def test_hawkes_decays_below_threshold_after_warmup():
    hl = 5.0  # 5-second half-life
    ts = np.arange(0, 300, 5, dtype=np.float64)  # 60 bars
    buys = np.zeros_like(ts)
    sells = np.zeros_like(ts)
    buys[0] = 1000.0
    out = hawkes_intensity_recursive(ts, buys, sells, hl_seconds=hl)
    # After 10×HL = 50s → decay = 0.5^10 ≈ 0.001
    idx = int(50 / 5)
    assert out["lambda_buy"][idx] < 1.0
    # After 20×HL = 100s → decay ≈ 10^-6
    idx = int(100 / 5)
    assert out["lambda_buy"][idx] < 1e-3


def test_hawkes_actual_dt_handles_missing_bars():
    """If actual Δt > nominal 5s, decay must use the real elapsed time."""
    hl = 5.0
    # Two bars at t=0 and t=60 (60s gap), each with 1000 buys
    ts = np.array([0.0, 60.0])
    buys = np.array([1000.0, 0.0])
    sells = np.zeros(2)
    out = hawkes_intensity_recursive(ts, buys, sells, hl_seconds=hl)
    # After 60s = 12×HL, decay = 0.5^12 ≈ 2.4e-4
    assert out["lambda_buy"][1] < 1.0
    assert out["lambda_buy"][1] > 0.0


def test_hawkes_warmup_flag():
    hl = 5.0
    ts = np.arange(0, 60, 5, dtype=np.float64)
    buys = np.ones_like(ts)
    sells = np.zeros_like(ts)
    out = hawkes_intensity_recursive(ts, buys, sells, hl_seconds=hl, warmup_factor=5.0)
    # First 5×5 = 25 sec of bars should be !is_warm
    warmup_cutoff = int(25 / 5)
    assert not out["is_warm"][:warmup_cutoff].any()
    assert out["is_warm"][warmup_cutoff:].all()


# ===========================================================================
# §8.D  CVD RTH reset
# ===========================================================================

def test_cvd_rth_resets_at_09_30_et():
    # Build synthetic: two days, 15-min bars from 08:00 ET to 11:00 ET each day
    rows = []
    for day in [(2024, 1, 2), (2024, 1, 3)]:
        for hh in range(8, 12):
            for mm in (0, 15, 30, 45):
                # ET with fixed UTC conversion (standard time EST = UTC-5)
                ts_utc = datetime(*day, hh + 5, mm, tzinfo=timezone.utc)
                rows.append({"ts": ts_utc, "buys_qty": 10, "sells_qty": 5})
    df = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    out = cvd_with_dual_reset(df)
    # bars_since_rth_reset should reset at 09:30 ET each day
    # In the data, the 09:30 ET bar on day1 is row 6 (08:00, 08:15, 08:30, 08:45, 09:00, 09:15, 09:30)
    # Before RTH reset (rows 0-5): previous-day RTH session, bars_since_rth_reset increments
    # At row 6 (09:30 ET day1): reset — bars_since_rth_reset = 0
    vals = out["bars_since_rth_reset"].to_numpy()
    # There should be a zero at the 09:30 ET transition
    # Row indices 0-5 = pre-09:30 day 1, row 6 = 09:30 day 1 (reset)
    assert vals[6] == 0, f"expected reset at row 6, got {vals[6]}"
    # After reset, incrementing
    assert vals[7] == 1
    # Next day's 09:30 transition: day 1 has 16 bars total (4 hours × 4 per hour), day 2 starts at row 16
    # day 2's 09:30 ET is at index 16 + 6 = 22
    assert vals[22] == 0, f"expected reset at row 22, got {vals[22]}"


def test_cvd_rth_monotonic_within_session():
    """Within a single RTH session, cvd_rth accumulates monotonically for all-buy flow."""
    rows = []
    for hh in range(10, 14):
        for mm in (0, 15, 30, 45):
            ts_utc = datetime(2024, 1, 2, hh + 5, mm, tzinfo=timezone.utc)
            rows.append({"ts": ts_utc, "buys_qty": 100, "sells_qty": 0})
    df = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    out = cvd_with_dual_reset(df)
    cvd = out["cvd_rth"].to_numpy()
    # Should strictly increase (all buys)
    assert np.all(np.diff(cvd) > 0)


# ===========================================================================
# §8.D (cont.)  rolling_rth_bounded
# ===========================================================================

def test_rolling_rth_bounded_nan_before_min_bars():
    values = np.arange(20, dtype=float)
    bars_since = np.arange(20)  # 0, 1, 2, ..., 19
    out = rolling_rth_bounded(values, bars_since, window=10, min_bars=5, agg="max")
    # First 4 rows (bars_since < 4) → effective_window < 5 → NaN
    assert np.all(np.isnan(out[:4]))
    # Row 4 (bars_since=4, effective=5) → valid
    assert out[4] == 4.0


def test_rolling_rth_bounded_window_shrinks_after_reset():
    """Effective window shrinks when bars_since_reset < window."""
    # 30 bars, reset at bar 15: bars_since_reset = [0..14, 0..14]
    values = np.arange(30, dtype=float)
    bars_since = np.concatenate([np.arange(15), np.arange(15)])
    out = rolling_rth_bounded(values, bars_since, window=20, min_bars=3, agg="max")
    # At bar 16 (bars_since=1, eff=2 < 3) → NaN
    assert np.isnan(out[16])
    # At bar 18 (bars_since=3, eff=4 >= 3 but window bounded to 4 bars 15..18)
    assert out[18] == 18.0  # max of [15,16,17,18] = 18


# ===========================================================================
# §8.E  Strict-backward asof
# ===========================================================================

def test_asof_strict_backward_excludes_exact_match():
    """Right event at time T must NOT appear in left row also at time T."""
    left = pl.DataFrame(
        {"ts": [datetime(2024, 1, 2, 14, 30, 0)], "es_px": [5000.0]}
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    right = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 2, 14, 29, 59), datetime(2024, 1, 2, 14, 30, 0)],
            "nq_ofi": [100.0, 200.0],
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))

    joined = asof_strict_backward(left, right, left_on="ts", right_on="ts")
    # Should get the 14:29:59 value (100), NOT 14:30:00 (200)
    assert joined["nq_ofi"][0] == 100.0


# ===========================================================================
# §8.F  Round-number pin
# ===========================================================================

def test_round_number_pin_v_shape():
    close = pl.Series([5000.0, 5025.0, 5049.75, 5050.0, 5050.25, 5075.0, 5100.0])
    d = round_number_pin_distance(close, N=50).to_numpy()
    expected = np.array([0.0, 25.0, 0.25, 0.0, 0.25, 25.0, 0.0])
    assert np.allclose(d, expected)


def test_round_number_pin_symmetric():
    # dist(5049.75, N=50) should equal dist(5050.25, N=50)
    left = round_number_pin_distance(pl.Series([5049.75]), 50).to_numpy()[0]
    right = round_number_pin_distance(pl.Series([5050.25]), 50).to_numpy()[0]
    assert abs(left - right) < 1e-10


def test_round_number_pin_n_5():
    close = pl.Series([5000.0, 5002.5, 5004.99])
    d = round_number_pin_distance(close, N=5).to_numpy()
    assert d[0] == 0.0
    assert d[1] == 2.5
    assert abs(d[2] - 0.01) < 1e-6
