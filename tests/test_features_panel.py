"""Tests for src/features/panel.py — built piece-by-piece.

Each step is tested independently before composing into the full panel pipeline.
"""
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

from src.features import panel


def _mk_phase_a_bars(n_days: int = 5, seed: int = 0) -> pl.DataFrame:
    """Build synthetic Phase A 15-min bars: ts + OHLCV + buys/sells + L1 close + CVD.

    Mimics what `src/data/bars_5sec.build_5sec_bars_core` + downsample produces.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 96
    rows = []
    base = datetime(2024, 3, 4, 0, 0, tzinfo=timezone.utc)  # Monday in EST
    px = 5000.0
    cvd = 0.0
    for d in range(n_days):
        for b in range(bars_per_day):
            ts = base + timedelta(days=d, minutes=15 * b)
            move = rng.normal(0, 1.5)
            px += move
            high = px + abs(rng.normal(0, 0.7))
            low = px - abs(rng.normal(0, 0.7))
            volume = max(1, int(rng.normal(1000, 300)))
            buys = int(volume * (0.5 + rng.normal(0, 0.05)))
            sells = max(1, volume - buys)
            cvd += (buys - sells) / 100.0
            mid = (px + 0.05 + px - 0.05) / 2  # mid ~ close
            spread = 0.25 + max(0, rng.normal(0, 0.05))
            rows.append(dict(
                ts=ts, open=px, high=high, low=low, close=px, volume=volume,
                dollar_volume=px * volume,
                buys_qty=buys, sells_qty=sells, trades_count=volume // 5,
                bid_close=mid - spread / 2, ask_close=mid + spread / 2,
                mid_close=mid, spread_abs_close=spread,
                cvd_globex=cvd,
            ))
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


# ---------------------------------------------------------------------------
# Step 1: attach_base_microstructure_features
# ---------------------------------------------------------------------------

def test_step1_base_features_emits_all_expected_columns():
    bars = _mk_phase_a_bars(n_days=5)
    out = panel.attach_base_microstructure_features(bars)
    for c in panel.BASE_VALUE_COLS:
        assert c in out.columns, f"missing base value column: {c}"


def test_step1_log_return_correctness():
    """log_return at row i should equal log(close[i] / close[i-1])."""
    bars = _mk_phase_a_bars(n_days=2)
    out = panel.attach_base_microstructure_features(bars)
    closes = bars["close"].to_list()
    expected_lr = [None] + [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    actual = out["log_return"].to_list()
    # First is null
    assert actual[0] is None
    # Remaining match
    for i in range(1, len(closes)):
        assert abs(actual[i] - expected_lr[i]) < 1e-9


def test_step1_ofi_bounds_minus_one_to_one():
    """OFI = (buys − sells) / (buys + sells) ∈ [-1, 1]."""
    bars = _mk_phase_a_bars(n_days=3)
    out = panel.attach_base_microstructure_features(bars)
    valid = out["ofi"].drop_nulls().to_list()
    assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in valid)


def test_step1_aggressor_ratio_bounds_zero_to_one():
    bars = _mk_phase_a_bars(n_days=3)
    out = panel.attach_base_microstructure_features(bars)
    valid = out["aggressor_ratio"].drop_nulls().to_list()
    assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in valid)


def test_step1_spread_to_mid_bps_positive():
    bars = _mk_phase_a_bars(n_days=3)
    out = panel.attach_base_microstructure_features(bars)
    valid = out["spread_to_mid_bps"].drop_nulls().to_list()
    assert all(v > 0 for v in valid)


def test_step1_realized_vol_warmup():
    """Realized vol with window=20 should be null for first 20 bars (after log_return)."""
    bars = _mk_phase_a_bars(n_days=3)
    out = panel.attach_base_microstructure_features(bars, rv_window=20)
    rv = out["realized_vol_w20"].to_list()
    # First ~20 should be null
    assert rv[0] is None
    # Later values should be positive (rolling std of returns)
    later = [v for v in rv[25:] if v is not None]
    assert len(later) > 0
    assert all(v >= 0 for v in later)


def test_step1_cvd_change_first_is_null():
    bars = _mk_phase_a_bars(n_days=2)
    out = panel.attach_base_microstructure_features(bars)
    cvd_chg = out["cvd_change"].to_list()
    assert cvd_chg[0] is None
    # Subsequent values are non-null
    later = cvd_chg[1:5]
    assert all(v is not None for v in later)


# ---------------------------------------------------------------------------
# Step 2: attach_ts_normalizations
# ---------------------------------------------------------------------------

def test_step2_emits_tc_and_madz_per_value_per_window():
    bars = _mk_phase_a_bars(n_days=40)  # enough for 30-day lookback
    df = panel.attach_base_microstructure_features(bars)
    df = panel.attach_ts_normalizations(
        df, value_cols=["log_return", "log_volume"], lookback_days_grid=(20, 30),
    )
    for v in ["log_return", "log_volume"]:
        for lb in (20, 30):
            assert f"{v}_tc_z_w{lb}" in df.columns
            assert f"{v}_tc_madz_w{lb}" in df.columns


def test_step2_skips_missing_columns_silently():
    """Pass a value_col that's not in the frame; function should skip without error."""
    bars = _mk_phase_a_bars(n_days=10)
    df = panel.attach_base_microstructure_features(bars)
    df = panel.attach_ts_normalizations(
        df, value_cols=["does_not_exist"], lookback_days_grid=(10,),
    )
    # No new columns added for the missing input
    assert "does_not_exist_tc_z_w10" not in df.columns


def test_step2_normalizations_match_underlying_primitives():
    """The TC z-score column from panel matches what tc_features.attach_tc_zscore produces directly."""
    from src.features.tc_features import attach_tc_zscore
    bars = _mk_phase_a_bars(n_days=40)
    df = panel.attach_base_microstructure_features(bars)
    df_panel = panel.attach_ts_normalizations(df, value_cols=["log_return"], lookback_days_grid=(20,))
    df_direct = attach_tc_zscore(df, value_col="log_return", lookback_days=20, out_col="check_z")
    # Values should match
    a = df_panel["log_return_tc_z_w20"].to_list()
    b = df_direct["check_z"].to_list()
    matched = 0
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        assert abs(x - y) < 1e-9
        matched += 1
    assert matched > 100, "too few comparable rows; tests not meaningfully checking equivalence"


# ---------------------------------------------------------------------------
# Step 3: build_wide_cross_asset_frame
# ---------------------------------------------------------------------------

def test_step3_wide_join_produces_prefixed_columns():
    a = _mk_phase_a_bars(n_days=3, seed=1)
    b = _mk_phase_a_bars(n_days=3, seed=2)
    a = panel.attach_base_microstructure_features(a)
    b = panel.attach_base_microstructure_features(b)
    wide = panel.build_wide_cross_asset_frame(
        {"ES": a, "NQ": b},
        base_value_cols=["log_return", "log_volume"],
    )
    for instr in ("ES", "NQ"):
        for v in ("log_return", "log_volume"):
            assert f"{instr}_{v}" in wide.columns


def test_step3_wide_join_outer_handles_missing_ts():
    """If two frames have non-overlapping ts, outer-join keeps both with nulls."""
    a = _mk_phase_a_bars(n_days=2, seed=1)
    b = _mk_phase_a_bars(n_days=2, seed=2).filter(
        pl.col("ts").dt.day() != pl.col("ts").dt.day().min()
    )
    a = panel.attach_base_microstructure_features(a)
    b = panel.attach_base_microstructure_features(b)
    wide = panel.build_wide_cross_asset_frame(
        {"ES": a, "NQ": b}, base_value_cols=["log_return"],
    )
    assert wide.height >= max(a.height, b.height)


# ---------------------------------------------------------------------------
# Step 4: attach_cross_sectional_ranks
# ---------------------------------------------------------------------------

def test_step4_cs_universe_rank_added_per_value():
    a = _mk_phase_a_bars(n_days=3, seed=1)
    b = _mk_phase_a_bars(n_days=3, seed=2)
    a = panel.attach_base_microstructure_features(a)
    b = panel.attach_base_microstructure_features(b)
    wide = panel.build_wide_cross_asset_frame(
        {"ES": a, "NQ": b}, base_value_cols=["log_return"],
    )
    out = panel.attach_cross_sectional_ranks(
        wide, base_value_cols=["log_return"], instruments=["ES", "NQ"],
    )
    for instr in ("ES", "NQ"):
        assert f"cs_universe_{instr}_log_return" in out.columns


# ---------------------------------------------------------------------------
# End-to-end (steps 1 + 2 via convenience function)
# ---------------------------------------------------------------------------

def test_per_instrument_pipeline_end_to_end():
    """build_per_instrument_features = step 1 + step 2 in one call."""
    bars = _mk_phase_a_bars(n_days=40)
    out = panel.build_per_instrument_features(
        bars, lookback_days_grid=(20, 30),
    )
    # Should have base cols + normalized variants
    for c in panel.BASE_VALUE_COLS:
        assert c in out.columns
        assert f"{c}_tc_z_w20" in out.columns
        assert f"{c}_tc_madz_w30" in out.columns
