"""Tests for src/features/cross_asset_macro.py."""
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

from src.features.cross_asset_macro import (
    DXY_WEIGHTS,
    attach_gauss_rank_cs,
    attach_mad_zscore,
    attach_rates_curve_spreads,
    attach_risk_on_off_composite,
    attach_rolling_correlation,
    attach_synthetic_dxy_logret,
)


def _mk_15min_bars(n_days: int, value_by_hour_utc: dict[int, float], rng_seed: int = 0):
    """Build n_days × 96 15-min bars with a 'value' column."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    base = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    for d in range(n_days):
        for b in range(96):
            ts = base + timedelta(days=d, minutes=15 * b)
            base_v = value_by_hour_utc.get(ts.hour, 1.0)
            v = base_v + rng.normal(0, 0.01)
            rows.append(dict(ts=ts, value=v))
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC"))).sort("ts")


# ---------------------------------------------------------------------------
# attach_mad_zscore
# ---------------------------------------------------------------------------

def test_mad_zscore_constant_per_hour_gives_near_zero():
    """If value at each hour is constant across days, MAD z → ~0 (after warmup)."""
    bars = _mk_15min_bars(n_days=40, value_by_hour_utc={h: float(h) for h in range(24)})
    out = attach_mad_zscore(bars, "value", lookback_days=20)
    valid = out.filter(pl.col("value_tc_madz").is_finite())
    assert valid.height > 0
    # Tiny noise → MAD ~ 0.01 → z within ±a few at most
    assert valid["value_tc_madz"].abs().max() < 10.0


def test_mad_zscore_marks_tail_event_with_large_z():
    """An outlier at one bar-of-day should produce a large MAD z-score for that bar.

    MAD z is robust to outliers in the SCALE estimate (median absolute deviation
    isn't inflated by a single outlier the way std is). So a single point at
    100x the typical value should produce a very large MAD z-score for itself,
    and shouldn't compress z-scores at other dates.
    """
    rows = []
    rng = np.random.default_rng(42)
    base = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    for d in range(40):
        for b in range(96):
            ts = base + timedelta(days=d, minutes=15 * b)
            v = 1.0 + rng.normal(0, 0.02)
            rows.append(dict(ts=ts, value=v, _d=d, _h=ts.hour, _m=ts.minute))
    bars = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC"))).sort("ts")
    # Inject a tail spike at d=35, hour=14, minute=0 (deep into the series, well past warmup)
    bars = bars.with_columns(
        pl.when((pl.col("_d") == 35) & (pl.col("_h") == 14) & (pl.col("_m") == 0))
        .then(pl.lit(5.0))
        .otherwise(pl.col("value"))
        .alias("value")
    ).drop(["_d", "_h", "_m"])

    out = attach_mad_zscore(bars, "value", lookback_days=20)
    spike = out.filter(
        (pl.col("ts").dt.hour() == 14)
        & (pl.col("ts").dt.minute() == 0)
        & (pl.col("value") > 4.0)
    )
    assert spike.height >= 1
    z = float(spike["value_tc_madz"][0])
    assert abs(z) > 20.0, f"MAD z for 5x-typical value should be very large; got {z}"


def test_mad_zscore_invalid_partition_minutes_raises():
    bars = _mk_15min_bars(n_days=2, value_by_hour_utc={})
    with pytest.raises(ValueError, match="partition_minutes"):
        attach_mad_zscore(bars, "value", partition_minutes=7)


# ---------------------------------------------------------------------------
# attach_gauss_rank_cs
# ---------------------------------------------------------------------------

def test_gauss_rank_cs_shape_and_columns():
    """Output should have one new column per input value_col."""
    panel = pl.DataFrame({
        "ts": [datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
               datetime(2024, 1, 2, 14, 15, tzinfo=timezone.utc)],
        "ret_a": [0.5, 0.1],
        "ret_b": [0.2, 0.3],
        "ret_c": [-0.1, 0.05],
    })
    out = attach_gauss_rank_cs(panel, ["ret_a", "ret_b", "ret_c"])
    for c in ["gauss_rank_ret_a", "gauss_rank_ret_b", "gauss_rank_ret_c"]:
        assert c in out.columns
    assert out.height == panel.height


def test_gauss_rank_cs_max_value_is_highest():
    """The largest value in a row gets the highest gauss rank in that row."""
    panel = pl.DataFrame({
        "a": [0.1, 0.5, 0.2],
        "b": [0.5, 0.1, 0.4],
        "c": [0.3, 0.2, 0.1],
    })
    out = attach_gauss_rank_cs(panel, ["a", "b", "c"])
    # Row 0: a=0.1, b=0.5, c=0.3 → b is largest → gauss_rank_b is largest
    r0 = out.row(0, named=True)
    vals_0 = {k: r0[f"gauss_rank_{k}"] for k in "abc"}
    assert max(vals_0, key=vals_0.get) == "b"
    # Row 1: a=0.5, b=0.1, c=0.2 → a is largest
    r1 = out.row(1, named=True)
    vals_1 = {k: r1[f"gauss_rank_{k}"] for k in "abc"}
    assert max(vals_1, key=vals_1.get) == "a"


def test_gauss_rank_cs_handles_nan():
    """NaN values in a row are skipped; ranking only uses valid values."""
    panel = pl.DataFrame({
        "a": [0.1, 0.5, float("nan")],
        "b": [0.5, float("nan"), 0.4],
        "c": [float("nan"), 0.2, 0.1],
    })
    out = attach_gauss_rank_cs(panel, ["a", "b", "c"])
    # Row 2 only has b=0.4 and c=0.1 valid → b is highest of 2
    r2 = out.row(2, named=True)
    assert r2["gauss_rank_b"] > r2["gauss_rank_c"]
    assert r2["gauss_rank_a"] is None or np.isnan(r2["gauss_rank_a"])


# ---------------------------------------------------------------------------
# attach_synthetic_dxy_logret
# ---------------------------------------------------------------------------

def test_synthetic_dxy_logret_weights_sum_to_one():
    s = sum(DXY_WEIGHTS.values())
    assert abs(s - 1.0) < 1e-9


def test_synthetic_dxy_logret_negative_when_all_currencies_strengthen():
    """All foreign currencies up vs USD → CME futures up → synthetic DXY DOWN (USD weak)."""
    panel = pl.DataFrame({
        "lr_eur": [0.01, -0.01],
        "lr_jpy": [0.01, -0.01],
        "lr_gbp": [0.01, -0.01],
        "lr_cad": [0.01, -0.01],
    })
    out = attach_synthetic_dxy_logret(panel, "lr_eur", "lr_jpy", "lr_gbp", "lr_cad")
    # Row 0: all currencies up → synthetic DXY logret < 0 (USD weakened)
    assert out["synthetic_dxy_logret"][0] < 0
    # Row 1: all currencies down → synthetic DXY logret > 0 (USD strengthened)
    assert out["synthetic_dxy_logret"][1] > 0


def test_synthetic_dxy_logret_is_weighted():
    """Output magnitude should equal the (renormalized) weighted sum."""
    panel = pl.DataFrame({
        "lr_eur": [0.01], "lr_jpy": [0.0], "lr_gbp": [0.0], "lr_cad": [0.0],
    })
    out = attach_synthetic_dxy_logret(panel, "lr_eur", "lr_jpy", "lr_gbp", "lr_cad")
    expected = -DXY_WEIGHTS["EUR"] * 0.01
    assert abs(out["synthetic_dxy_logret"][0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# attach_rates_curve_spreads
# ---------------------------------------------------------------------------

def test_rates_curve_spreads_basic():
    panel = pl.DataFrame({
        "lr_zt": [0.001],   # 2y futures up 10bp
        "lr_zf": [0.002],   # 5y up 20bp
        "lr_zn": [0.003],   # 10y up 30bp
        "lr_zb": [0.004],   # 30y up 40bp
    })
    out = attach_rates_curve_spreads(panel, "lr_zt", "lr_zf", "lr_zn", "lr_zb")
    assert abs(out["slope_2s5s_logret"][0] - 0.001) < 1e-9
    assert abs(out["slope_5s10s_logret"][0] - 0.001) < 1e-9
    assert abs(out["slope_2s10s_logret"][0] - 0.002) < 1e-9
    assert abs(out["slope_10s30s_logret"][0] - 0.001) < 1e-9
    # Butterfly = ZN − 2·ZF + ZT = 0.003 − 0.004 + 0.001 = 0.0
    assert abs(out["butterfly_2s5s10s"][0]) < 1e-9


# ---------------------------------------------------------------------------
# attach_rolling_correlation
# ---------------------------------------------------------------------------

def test_rolling_correlation_perfect_positive():
    """Two identical sequences → rolling corr = 1.0 (after warmup)."""
    n = 30
    panel = pl.DataFrame({
        "a": [float(i) for i in range(n)],
        "b": [float(i) for i in range(n)],
    })
    out = attach_rolling_correlation(panel, "a", "b", window=10)
    valid = out.filter(pl.col("corr_a_b_w10").is_not_null())
    assert all(abs(v - 1.0) < 1e-6 for v in valid["corr_a_b_w10"].to_list())


def test_rolling_correlation_perfect_negative():
    n = 30
    panel = pl.DataFrame({
        "a": [float(i) for i in range(n)],
        "b": [float(-i) for i in range(n)],
    })
    out = attach_rolling_correlation(panel, "a", "b", window=10)
    valid = out.filter(pl.col("corr_a_b_w10").is_not_null())
    assert all(abs(v - (-1.0)) < 1e-6 for v in valid["corr_a_b_w10"].to_list())


# ---------------------------------------------------------------------------
# attach_risk_on_off_composite
# ---------------------------------------------------------------------------

def test_risk_on_off_composite_high_when_risk_off():
    """Gold↑ + bonds↑ + DXY↑ + equities↓ → high score (risk-off)."""
    panel = pl.DataFrame({
        "gold_z": [2.0],
        "dxy_z":  [2.0],
        "zn_z":   [2.0],
        "zb_z":   [2.0],
        "es_z":   [-2.0],
        "nq_z":   [-2.0],
    })
    out = attach_risk_on_off_composite(
        panel, "gold_z", "dxy_z", ["zn_z", "zb_z"], ["es_z", "nq_z"],
    )
    score = float(out["risk_off_score"][0])
    # gold + mean(bonds) + dxy - mean(equities) = 2 + 2 + 2 - (-2) = 8
    assert abs(score - 8.0) < 1e-9


def test_risk_on_off_composite_negative_when_risk_on():
    """Gold↓ + bonds↓ + DXY↓ + equities↑ → low/negative score (risk-on)."""
    panel = pl.DataFrame({
        "gold_z": [-1.0],
        "dxy_z":  [-1.0],
        "zn_z":   [-1.0],
        "zb_z":   [-1.0],
        "es_z":   [1.0],
        "nq_z":   [1.0],
    })
    out = attach_risk_on_off_composite(
        panel, "gold_z", "dxy_z", ["zn_z", "zb_z"], ["es_z", "nq_z"],
    )
    score = float(out["risk_off_score"][0])
    # -1 + (-1) + (-1) - 1 = -4
    assert abs(score - (-4.0)) < 1e-9
