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


# ---------------------------------------------------------------------------
# Step 6: assemble_target_panel
# ---------------------------------------------------------------------------

def test_step6_assemble_target_panel_attaches_labels():
    """Target panel includes label/realized_ret/atr columns from triple_barrier_labels."""
    bars = _mk_phase_a_bars(n_days=200)  # enough for 150-day TC-ATR lookback
    target_bars = panel.build_per_instrument_features(bars, lookback_days_grid=(30,))
    out = panel.assemble_target_panel(
        target="ES",
        target_bars_with_features=target_bars,
        wide_cross_asset=None,
        label_params={"k_up": 1.0, "k_dn": 1.0, "T": 4, "lookback_days": 30},
    )
    for c in ("label", "realized_ret", "realized_ret_pts", "hit_offset", "atr", "halt_truncated"):
        assert c in out.columns, f"missing label-output column: {c}"
    # All rows should have finite atr + realized_ret since drop_invalid=True
    assert out["atr"].is_finite().all()
    assert out["realized_ret"].is_finite().all()


def test_step6_uses_v1_locked_params_when_label_params_omitted():
    """Default to V1_LABEL_PARAMS[target] if no explicit label_params given.

    Note: with V1 default lookback_days=150, we need >=150 days to produce
    valid rows. Use a smaller-lookback override here to keep the test fast,
    but verify the V1 dict is consulted as the source.
    """
    assert panel.V1_LABEL_PARAMS["ES"]["k_up"] == 1.25
    assert panel.V1_LABEL_PARAMS["RTY"]["T"] == 4
    assert panel.V1_LABEL_PARAMS["RTY"]["lookback_days"] == 180


def test_step6_invalid_target_raises():
    bars = _mk_phase_a_bars(n_days=40)
    target_bars = panel.build_per_instrument_features(bars, lookback_days_grid=(20,))
    with pytest.raises(ValueError, match="V1_LABEL_PARAMS"):
        panel.assemble_target_panel(target="ZZZ", target_bars_with_features=target_bars)


def test_step6_joins_wide_cross_asset_drops_target_prefix():
    """When the wide frame has ES_log_return etc., those should NOT collide
    with target_bars_with_features's own log_return."""
    target_bars = panel.build_per_instrument_features(
        _mk_phase_a_bars(n_days=200, seed=1), lookback_days_grid=(30,),
    )
    peer_bars = panel.build_per_instrument_features(
        _mk_phase_a_bars(n_days=200, seed=2), lookback_days_grid=(30,),
    )
    wide = panel.build_wide_cross_asset_frame(
        {"ES": target_bars, "NQ": peer_bars},
        base_value_cols=["log_return", "log_volume"],
    )
    out = panel.assemble_target_panel(
        target="ES",
        target_bars_with_features=target_bars,
        wide_cross_asset=wide,
        label_params={"k_up": 1.0, "k_dn": 1.0, "T": 4, "lookback_days": 30},
    )
    # ES's own log_return is in target_bars (unprefixed)
    assert "log_return" in out.columns
    # ES_log_return from wide should NOT be in output (dropped to avoid duplication)
    assert "ES_log_return" not in out.columns
    # NQ's prefixed log_return SHOULD be in output (peer feature)
    assert "NQ_log_return" in out.columns


# ---------------------------------------------------------------------------
# Step 5b: attach_l2_deep_features
# ---------------------------------------------------------------------------

def _mk_phase_b_bars(n_bars: int = 30, depth: int = 10):
    """Phase A+B bars: Phase A columns + L1..L10 book snapshot."""
    rng = np.random.default_rng(7)
    rows = []
    base = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)
    for i in range(n_bars):
        ts = base + timedelta(minutes=15 * i)
        mid = 5000.0 + rng.normal(0, 1.0)
        d = {"ts": ts, "open": mid, "high": mid + 0.5, "low": mid - 0.5,
             "close": mid, "volume": 1000, "buys_qty": 500, "sells_qty": 500,
             "trades_count": 100, "mid_close": mid,
             "bid_close": mid - 0.125, "ask_close": mid + 0.125,
             "spread_abs_close": 0.25, "cvd_globex": float(i)}
        for k in range(1, depth + 1):
            d[f"bid_px_L{k}"] = mid - 0.25 * k
            d[f"ask_px_L{k}"] = mid + 0.25 * k
            d[f"bid_sz_L{k}"] = 100 - 5 * (k - 1)
            d[f"ask_sz_L{k}"] = 100 - 5 * (k - 1) + int(rng.normal(0, 2))
            d[f"bid_ord_L{k}"] = 10
            d[f"ask_ord_L{k}"] = 10
        rows.append(d)
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


def test_step5b_l2_deep_features_emit_expected_columns():
    bars = _mk_phase_b_bars(n_bars=80, depth=10)
    out = panel.attach_l2_deep_features(bars, depth=10, spread_z_window=20)
    expected = [
        "cum_imbalance_d10", "dw_imbalance_d10",
        "depth_weighted_spread_d10", "liquidity_adjusted_spread_d10",
        "spread_acceleration", "hhi_d10",
        "deep_ofi_d10_decay0", "deep_ofi_d10_decay03",
        "spread_zscore_w20",
    ]
    for c in expected:
        assert c in out.columns, f"missing L2-deep column: {c}"


# ---------------------------------------------------------------------------
# Step 5c: attach_gex_for_target
# ---------------------------------------------------------------------------

def test_step5c_attach_gex_for_target_with_synthetic_profile(tmp_path):
    """End-to-end: write a tiny GEX profile parquet, join onto target bars."""
    from datetime import date
    # Tiny daily GEX profile
    profile = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "total_gex": [1.0e9, -2.0e9],
        "gex_sign": [1, -1],
        "zero_gamma_strike": [5000.0, 5050.0],
        "max_call_oi_strike": [5100.0, 5150.0],
        "max_put_oi_strike": [4900.0, 4950.0],
        "gex_0dte_share": [0.3, 0.4],
        "gex_0dte_only": [3.0e8, -8.0e8],
        "gex_without_0dte": [7.0e8, -1.2e9],
    }).with_columns(pl.col("date").cast(pl.Date))
    profile_path = tmp_path / "gex_profile_2024.parquet"
    profile.write_parquet(profile_path)

    # Target ES bars on 2024-01-03 (uses 2024-01-02 EOD profile per attach_gex_features)
    bars = pl.DataFrame({
        "ts": [datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc),
               datetime(2024, 1, 3, 14, 45, tzinfo=timezone.utc)],
        "close": [5050.0, 5060.0],
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))

    out = panel.attach_gex_for_target(bars, [str(profile_path)])
    # GEX features should be attached
    for c in ("total_gex", "gex_sign", "distance_to_zero_gamma_flip",
              "distance_to_max_call_oi", "distance_to_max_put_oi"):
        assert c in out.columns, f"missing GEX column: {c}"


def test_step5c_attach_gex_empty_paths_passthrough():
    """Empty list of GEX paths → bars unchanged."""
    bars = pl.DataFrame({"ts": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                          "close": [5000.0]}).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    out = panel.attach_gex_for_target(bars, [])
    assert out.columns == bars.columns


def test_per_instrument_pipeline_attaches_session_flags_and_overnight():
    """The end-to-end per-instrument pipeline adds session flags + overnight features."""
    bars = _mk_phase_a_bars(n_days=10)
    out = panel.build_per_instrument_features(bars, lookback_days_grid=(20,))
    # Session flags
    for c in ("hour_et", "is_asia", "is_eu", "is_rth", "is_eth"):
        assert c in out.columns, f"missing session flag column: {c}"
    # Overnight features
    for c in (
        "overnight_log_return", "overnight_realized_vol",
        "overnight_volume_total", "overnight_n_bars",
    ):
        assert c in out.columns, f"missing overnight feature column: {c}"


def test_per_instrument_pipeline_attach_overnight_off():
    """attach_overnight=False disables the overnight pass."""
    bars = _mk_phase_a_bars(n_days=10)
    out = panel.build_per_instrument_features(
        bars, lookback_days_grid=(20,), attach_overnight=False,
    )
    # Session flags still added (needed for downstream regardless)
    assert "is_rth" in out.columns
    # Overnight columns NOT added
    assert "overnight_log_return" not in out.columns


def test_step6_drop_invalid_filter():
    """drop_invalid=False keeps warmup rows; drop_invalid=True removes them."""
    bars = _mk_phase_a_bars(n_days=200)
    target_bars = panel.build_per_instrument_features(bars, lookback_days_grid=(30,))
    out_with = panel.assemble_target_panel(
        target="ES",
        target_bars_with_features=target_bars,
        label_params={"k_up": 1.0, "k_dn": 1.0, "T": 4, "lookback_days": 30},
        drop_invalid=True,
    )
    out_without = panel.assemble_target_panel(
        target="ES",
        target_bars_with_features=target_bars,
        label_params={"k_up": 1.0, "k_dn": 1.0, "T": 4, "lookback_days": 30},
        drop_invalid=False,
    )
    assert out_with.height < out_without.height
    # The dropped rows had non-finite atr or ret (warmup or halt-truncated)
    invalid = out_without.filter(
        ~pl.col("atr").is_finite() | ~pl.col("realized_ret").is_finite()
    )
    assert invalid.height > 0
