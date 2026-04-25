"""Per-instrument feature computation — single-contract pipeline.

This module owns the bar-derived feature attach passes for a single contract.
It is data-source-agnostic; everything here operates on (Phase A or A+B) bar
frames and returns a frame with new feature columns appended.

Pipeline (orchestrated by `build_per_instrument_features`):
    Phase A bars
    → attach_base_microstructure_features    (returns/flow/vol/moments/illiq)
    → attach_session_flags + cyclic minute    (from src/features/tc_features)
    → attach_overnight_features               (from src/features/overnight)
    → attach_pattern_features                  (Tier-7 detectors)
    → attach_engine_features                  (fracdiff + round-pin)
    → attach_ts_normalizations                (TC z + MAD z × multiple lookback days)
    → attach_ema_smoothed                     (causal EMA-smoothed variants)

Phase A+B-only features (require L1-L10 book columns) live in
`attach_l2_deep_features` and are typically called by the panel orchestrator
after `build_per_instrument_features` finishes.
"""
from __future__ import annotations

import polars as pl

from . import bar as bar_features
from . import bar_agg
from . import cross_asset_macro
from . import deep_ofi as deep_ofi_features
from . import engines as engine_features
from . import l1 as l1_features
from . import l2 as l2_features
from . import overnight
from . import patterns as pattern_features
from . import smoothed as smoothed_features
from . import tc_features

EPS = 1e-9


# Base values computed per instrument that get TC+MAD z-scored AND
# (if a cross-sectional panel is built) Gauss-Ranked downstream.
BASE_VALUE_COLS = [
    # Core returns & flow
    "log_return", "abs_log_return", "log_volume",
    "ofi", "aggressor_ratio", "cvd_change", "net_aggressor_volume",
    "average_trade_size", "trades_count_log",
    # Implied volume (T1.39) — derived from Algoseek implied_* Phase A cols
    "implied_volume_share", "implied_aggressor_skew",
    # Spread
    "spread_to_mid_bps", "spread_sub_var_ratio",
    # Realized vol family
    "realized_vol_w20", "realized_vol_w60", "realized_vol_w120",
    "vol_surprise_w20", "vol_surprise_w60", "vol_surprise_w120",
    # Range vol family
    "range_vol_parkinson_w20", "range_vol_parkinson_w60", "range_vol_gk_w20",
    # Asymmetric vol + ratio
    "up_vol_w60", "down_vol_w60", "vol_direction_ratio_w60",
    "vol_ratio_short_long",
    # Higher moments
    "rolling_skew_w60", "rolling_kurt_w60", "vol_of_vol_w60",
    # Microstructure
    "amihud_illiq_w20", "return_autocorr_lag1_w60",
    "vwap_return_w20", "price_volume_corr_w60", "price_impact_slope_w60",
    "jump_indicator_w20", "vwap_deviation",
]


# ---------------------------------------------------------------------------
# Step 1 — per-instrument base microstructure features
# ---------------------------------------------------------------------------

def attach_base_microstructure_features(
    bars: pl.DataFrame,
    rv_windows: tuple[int, ...] = (20, 60, 120),
    range_vol_windows: tuple[int, ...] = (20, 60),
    vol_surprise_windows: tuple[int, ...] = (20, 60, 120),
    higher_moment_window: int = 60,
    asym_vol_window: int = 60,
    autocorr_window: int = 60,
    pvc_window: int = 60,
    pis_window: int = 60,
    amihud_window: int = 20,
    vwap_ret_window: int = 20,
    jump_window: int = 20,
    short_vol_w: int = 20,
    long_vol_w: int = 60,
) -> pl.DataFrame:
    """Compute the FULL set of base per-bar features for a single instrument.

    Phase A bars must include: ts, open, high, low, close, volume,
    buys_qty, sells_qty, trades_count, mid_close, spread_abs_close, cvd_globex,
    spread_mean_sub, spread_std_sub.

    Adds (in three .with_columns passes to respect dependency ordering):

    Pass 1 — pure transforms:
        log_return, abs_log_return, log_volume, trades_count_log
        ofi, aggressor_ratio, net_aggressor_volume, average_trade_size
        cvd_change, spread_to_mid_bps, spread_sub_var_ratio
        rolling_volume_sum_w20, rolling_volume_std_w20, rolling_trade_count_mean_w20

    Pass 2 — rolling stats on log_return / log_volume:
        realized_vol_w{20,60,120}, vol_surprise_w{20,60,120}
        range_vol_parkinson_w{20,60}, range_vol_gk_w20
        up_vol_w60, down_vol_w60
        rolling_skew_w60, rolling_kurt_w60
        amihud_illiq_w20, return_autocorr_lag1_w60
        vwap_return_w20, price_volume_corr_w60, price_impact_slope_w60
        jump_indicator_w20

    Pass 3 — derived ratios:
        vol_direction_ratio_w60 (= down_vol/up_vol)
        vol_of_vol_w60 (rolling std of realized_vol_w20)
        vol_ratio_short_long (= realized_vol_w20 / realized_vol_w60)
        vwap_deviation (close − rolling vwap proxy via VWAP-return integration)

    Returns the original bars DataFrame plus these feature columns.
    """
    cols = set(bars.columns)
    pass1 = [
        bar_features.log_return(pl.col("close")).alias("log_return"),
        bar_features.log_volume(pl.col("volume")).alias("log_volume"),
        bar_agg.order_flow_imbalance(pl.col("buys_qty"), pl.col("sells_qty")).alias("ofi"),
        bar_agg.aggressor_side_ratio(pl.col("buys_qty"), pl.col("sells_qty")).alias("aggressor_ratio"),
        bar_agg.net_aggressor_volume(pl.col("buys_qty"), pl.col("sells_qty")).alias("net_aggressor_volume"),
        (pl.col("cvd_globex") - pl.col("cvd_globex").shift(1)).alias("cvd_change"),
        (pl.col("spread_abs_close") / (pl.col("mid_close") + EPS) * 10_000).alias("spread_to_mid_bps"),
        bar_agg.rolling_volume_sum(pl.col("volume"), window=20).alias("rolling_volume_sum_w20"),
        bar_agg.rolling_volume_std(pl.col("volume"), window=20).alias("rolling_volume_std_w20"),
    ]
    if "trades_count" in cols:
        pass1.append(
            bar_agg.average_trade_size(pl.col("volume"), pl.col("trades_count")).alias("average_trade_size")
        )
        pass1.append((1 + pl.col("trades_count")).log().alias("trades_count_log"))
        pass1.append(
            bar_agg.rolling_trade_count_mean(pl.col("trades_count"), window=20).alias("rolling_trade_count_mean_w20")
        )
    if {"spread_std_sub", "spread_mean_sub"}.issubset(cols):
        pass1.append(
            (pl.col("spread_std_sub") / (pl.col("spread_mean_sub") + EPS)).alias("spread_sub_var_ratio")
        )
    if {"implied_volume", "implied_buys", "implied_sells"}.issubset(cols):
        # T1.39 ImpliedVolumeShare: fraction of bar volume that's implied (CME spreader fills)
        pass1.append(
            (pl.col("implied_volume") / (pl.col("volume") + EPS)).alias("implied_volume_share")
        )
        # Within implied volume, what's the buy/sell skew? Range [-1, 1].
        pass1.append(
            ((pl.col("implied_buys") - pl.col("implied_sells"))
             / (pl.col("implied_volume") + EPS)).alias("implied_aggressor_skew")
        )
    df = bars.with_columns(pass1).with_columns([
        pl.col("log_return").abs().alias("abs_log_return"),
    ])

    expansions: list[pl.Expr] = []
    for w in rv_windows:
        expansions.append(
            bar_features.realized_volatility(pl.col("log_return"), window=w, method="std")
                .alias(f"realized_vol_w{w}")
        )
    for w in vol_surprise_windows:
        expansions.append(
            (pl.col("log_volume") - pl.col("log_volume").rolling_mean(window_size=w))
                .alias(f"vol_surprise_w{w}")
        )
    for w in range_vol_windows:
        expansions.append(
            bar_features.range_vol_parkinson(pl.col("high"), pl.col("low"), window=w)
                .alias(f"range_vol_parkinson_w{w}")
        )
    expansions.append(
        bar_features.range_vol_gk(
            pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"), window=20
        ).alias("range_vol_gk_w20")
    )
    expansions.append(
        l1_features.up_volatility(pl.col("log_return"), window=asym_vol_window)
            .alias(f"up_vol_w{asym_vol_window}")
    )
    expansions.append(
        l1_features.down_volatility(pl.col("log_return"), window=asym_vol_window)
            .alias(f"down_vol_w{asym_vol_window}")
    )
    expansions.append(
        l1_features.tick_return_skew(pl.col("log_return"), window=higher_moment_window)
            .alias(f"rolling_skew_w{higher_moment_window}")
    )
    expansions.append(
        l1_features.tick_return_kurtosis(pl.col("log_return"), window=higher_moment_window)
            .alias(f"rolling_kurt_w{higher_moment_window}")
    )
    expansions.append(
        bar_features.amihud_illiquidity(
            pl.col("log_return"), pl.col("volume"), pl.col("close"), window=amihud_window,
        ).alias(f"amihud_illiq_w{amihud_window}")
    )
    expansions.append(
        bar_features.return_autocorrelation(pl.col("log_return"), lag=1, window=autocorr_window)
            .alias(f"return_autocorr_lag1_w{autocorr_window}")
    )
    expansions.append(
        bar_features.vwap_return(
            pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"),
            window=vwap_ret_window,
        ).alias(f"vwap_return_w{vwap_ret_window}")
    )
    expansions.append(
        bar_features.price_volume_correlation(
            pl.col("log_return"), pl.col("log_volume"), window=pvc_window,
        ).alias(f"price_volume_corr_w{pvc_window}")
    )
    expansions.append(
        bar_features.price_impact_slope(
            pl.col("log_return"), pl.col("log_volume"), window=pis_window,
        ).alias(f"price_impact_slope_w{pis_window}")
    )
    expansions.append(
        bar_features.jump_indicator(
            pl.col("log_return"),
            bar_features.realized_volatility(pl.col("log_return"), window=jump_window, method="std"),
            threshold=3.0, output="zscore",
        ).alias(f"jump_indicator_w{jump_window}")
    )
    df = df.with_columns(expansions)

    pass3 = [
        l1_features.vol_direction_ratio(
            pl.col(f"down_vol_w{asym_vol_window}"),
            pl.col(f"up_vol_w{asym_vol_window}"),
        ).alias(f"vol_direction_ratio_w{asym_vol_window}"),
    ]
    have_short = short_vol_w in rv_windows
    have_long = long_vol_w in rv_windows
    if have_short:
        pass3.append(
            l1_features.volatility_of_volatility(
                pl.col(f"realized_vol_w{short_vol_w}"), window=long_vol_w,
            ).alias(f"vol_of_vol_w{long_vol_w}")
        )
    if have_short and have_long:
        pass3.append(
            bar_features.volatility_ratio(
                pl.col(f"realized_vol_w{short_vol_w}"), pl.col(f"realized_vol_w{long_vol_w}"),
            ).alias("vol_ratio_short_long")
        )
    df = df.with_columns(pass3)

    df = df.with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_et_date")
    )
    df = df.with_columns(
        (
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0 * pl.col("volume"))
                .cum_sum().over("_et_date")
            / (pl.col("volume").cum_sum().over("_et_date") + EPS)
        ).alias("_session_vwap_proxy")
    )
    df = df.with_columns(
        bar_features.vwap_deviation(pl.col("close"), pl.col("_session_vwap_proxy"))
            .alias("vwap_deviation")
    ).drop(["_et_date", "_session_vwap_proxy"])

    return df


# ---------------------------------------------------------------------------
# Step 2 — time-series normalizations (TC z + MAD z) at multiple windows
# ---------------------------------------------------------------------------

def attach_ts_normalizations(
    bars: pl.DataFrame,
    value_cols: list[str],
    lookback_days_grid: tuple[int, ...] = (30, 60),
    bar_minutes: int = 15,
    partition_minutes: int = 15,
    ts_col: str = "ts",
) -> pl.DataFrame:
    """For each value column × lookback_days, attach TC z-score AND MAD z-score.

    Output column naming:
        {value_col}_tc_z_w{lookback_days}
        {value_col}_tc_madz_w{lookback_days}
    """
    df = bars
    for value_col in value_cols:
        if value_col not in df.columns:
            continue
        for lb in lookback_days_grid:
            df = tc_features.attach_tc_zscore(
                df, value_col=value_col, lookback_days=lb,
                bar_minutes=bar_minutes, partition_minutes=partition_minutes,
                ts_col=ts_col,
                out_col=f"{value_col}_tc_z_w{lb}",
            )
            df = cross_asset_macro.attach_mad_zscore(
                df, value_col=value_col, lookback_days=lb,
                bar_minutes=bar_minutes, partition_minutes=partition_minutes,
                ts_col=ts_col,
                out_col=f"{value_col}_tc_madz_w{lb}",
            )
    return df


# ---------------------------------------------------------------------------
# L2-deep features (only on Phase A+B bars for trading instruments)
# ---------------------------------------------------------------------------

def attach_l2_deep_features(
    bars: pl.DataFrame,
    depth: int = 10,
    spread_z_window: int = 60,
) -> pl.DataFrame:
    """Attach L2-derived features to bars that have L1-L10 columns.

    Requires bid_px_L{1..depth}, ask_px_L{1..depth}, bid_sz_L{1..depth},
    ask_sz_L{1..depth} columns (output of `src/data/depth_snap.attach_book_snapshot`,
    Phase A+B bars).

    Adds composite (10) + per-level (40) features. See BUILD_PIPELINE.md
    column-inventory for the full list.
    """
    aggs = [
        l2_features.cumulative_imbalance(depth=depth).alias(f"cum_imbalance_d{depth}"),
        l2_features.distance_weighted_imbalance(depth=depth).alias(f"dw_imbalance_d{depth}"),
        l2_features.depth_weighted_spread(depth=depth).alias(f"depth_weighted_spread_d{depth}"),
        l2_features.liquidity_adjusted_spread(depth=depth).alias(f"liquidity_adjusted_spread_d{depth}"),
        l2_features.spread_acceleration().alias("spread_acceleration"),
        l2_features.herfindahl_hirschman_index(side="bid", depth=depth).alias(f"hhi_bid_d{depth}"),
        l2_features.herfindahl_hirschman_index(side="ask", depth=depth).alias(f"hhi_ask_d{depth}"),
        deep_ofi_features.deep_ofi(max_depth=depth, decay=0.0).alias(f"deep_ofi_d{depth}_decay0"),
        deep_ofi_features.deep_ofi(max_depth=depth, decay=0.3).alias(f"deep_ofi_d{depth}_decay03"),
        l2_features.spread_zscore(depth=1, window=spread_z_window).alias(f"spread_zscore_w{spread_z_window}"),
    ]
    for k in range(1, depth + 1):
        aggs.append(l2_features.volume_imbalance_at(k=k).alias(f"volume_imbalance_L{k}"))
        aggs.append(l2_features.basic_spread_at(k=k).alias(f"basic_spread_L{k}"))
        aggs.append(deep_ofi_features.ofi_at_level(k=k).alias(f"ofi_at_L{k}"))
        if k <= 5:
            aggs.append(l2_features.order_count_imbalance_at(k=k).alias(f"order_count_imbalance_L{k}"))
            aggs.append(l2_features.order_size_imbalance_at(k=k).alias(f"order_size_imbalance_L{k}"))
    return bars.with_columns(aggs)


# ---------------------------------------------------------------------------
# Pattern features (T7.* — absorption / breakouts / divergence)
# ---------------------------------------------------------------------------

def attach_pattern_features(
    bars: pl.DataFrame,
    lookback_breakout: int = 30,
    lookback_divergence: int = 30,
    spike_lookback: int = 20,
    persistence_window: int = 30,
    range_compression_window: int = 20,
    atr_proxy_window: int = 20,
) -> pl.DataFrame:
    """Attach pattern detectors (Tier-7) on top of base microstructure features.

    Requires bars to already contain: log_return, ofi, cvd_globex (+ cvd_rth
    optional), buys_qty, sells_qty, range_vol_parkinson_w20, close, volume.
    """
    atr_col = f"range_vol_parkinson_w{atr_proxy_window}"
    if atr_col not in bars.columns:
        bars = bars.with_columns(
            (pl.col("high") / pl.col("low")).log().rolling_mean(window_size=atr_proxy_window)
                .mul(pl.col("close")).alias("atr_proxy")
        )
        atr_expr = pl.col("atr_proxy")
    else:
        bars = bars.with_columns((pl.col(atr_col) * pl.col("close")).alias("atr_proxy"))
        atr_expr = pl.col("atr_proxy")

    aggs: list[pl.Expr] = [
        pattern_features.breakout_magnitude_up(
            pl.col("high"), atr_expr, lookback_bars=lookback_breakout,
        ).alias(f"breakout_magnitude_up_w{lookback_breakout}"),
        pattern_features.breakout_magnitude_down(
            pl.col("low"), atr_expr, lookback_bars=lookback_breakout,
        ).alias(f"breakout_magnitude_down_w{lookback_breakout}"),
        pattern_features.breakout_reversal_up(
            pl.col("high"), pl.col("close"), atr_expr,
            lookback_bars=lookback_breakout, reversal_atr=0.5,
        ).alias(f"breakout_reversal_up_w{lookback_breakout}"),
        pattern_features.breakout_reversal_down(
            pl.col("low"), pl.col("close"), atr_expr,
            lookback_bars=lookback_breakout, reversal_atr=0.5,
        ).alias(f"breakout_reversal_down_w{lookback_breakout}"),
        pattern_features.spike_and_fade_volume(
            pl.col("volume"), lookback_bars=spike_lookback, spike_multiplier=3.0,
        ).alias(f"spike_and_fade_volume_w{spike_lookback}"),
        pattern_features.imbalance_persistence_runlength(
            pl.col("ofi"), window=persistence_window,
        ).alias(f"imbalance_persistence_runlength_w{persistence_window}"),
        pattern_features.cvd_price_divergence_up(
            pl.col("cvd_globex"), pl.col("high"), window=lookback_divergence,
        ).alias(f"cvd_price_divergence_up_w{lookback_divergence}"),
        pattern_features.cvd_price_divergence_down(
            pl.col("cvd_globex"), pl.col("low"), window=lookback_divergence,
        ).alias(f"cvd_price_divergence_down_w{lookback_divergence}"),
        pattern_features.range_compression_ratio(
            pl.col("high"), pl.col("low"), atr_expr, window=range_compression_window,
        ).alias(f"range_compression_ratio_w{range_compression_window}"),
        pattern_features.absorption_score(
            ((pl.col("buys_qty") - pl.col("sells_qty")) * pl.col("close")),
            pl.col("realized_vol_w20") if "realized_vol_w20" in bars.columns
                else pl.col("log_return").rolling_std(window_size=20),
            (pl.col("close") * pl.col("volume")),
            window=20,
        ).alias("absorption_score_w20"),
    ]
    df = bars.with_columns(aggs)

    df = df.with_columns([
        pattern_features.post_breakout_flow_reversal(
            pl.col(f"breakout_reversal_up_w{lookback_breakout}"),
            pl.col("ofi"), lookforward_bars=3,
        ).alias("post_breakout_flow_reversal_up"),
        pattern_features.post_breakout_flow_reversal(
            pl.col(f"breakout_reversal_down_w{lookback_breakout}"),
            pl.col("ofi"), lookforward_bars=3,
        ).alias("post_breakout_flow_reversal_down"),
    ])
    return df


# ---------------------------------------------------------------------------
# Engine-level features (fracdiff, round_number_pin)
# ---------------------------------------------------------------------------

def attach_engine_features(
    bars: pl.DataFrame,
    fracdiff_d: float = 0.4,
    fracdiff_tau: float = 1e-5,
    round_pin_N_grid: tuple[float, ...] = (10.0, 25.0, 50.0),
    close_col: str = "close",
) -> pl.DataFrame:
    """Attach engine-level features: fixed-d FFD on log(close), round-number-pin distance."""
    log_close = bars.select(pl.col(close_col).log().alias("_lc"))["_lc"]
    fd = engine_features.fracdiff_series(log_close, d=fracdiff_d, tau=fracdiff_tau)
    df = bars.with_columns(pl.Series(f"fracdiff_logclose_d{fracdiff_d}", fd))
    for N in round_pin_N_grid:
        rn = engine_features.round_number_pin_distance(bars[close_col], N=float(N))
        df = df.with_columns(pl.Series(f"round_pin_distance_N{int(N)}", rn))
    return df


# ---------------------------------------------------------------------------
# Phase E features (execution-quality + cancel-proxy + quote-event derived ratios)
# ---------------------------------------------------------------------------

def attach_phase_e_features(
    bars: pl.DataFrame,
    phase_e_bars: pl.DataFrame,
    ts_col: str = "ts",
    hidden_absorption_window: int = 30,
) -> pl.DataFrame:
    """Asof-join Phase E aggregates onto the bar frame and compute derived ratios.

    Inputs:
        bars: per-bar futures panel (must have ts, volume, trades_count).
        phase_e_bars: per-bar Phase E aggregates produced by
            scripts/build_phase_e_bars.py — has eff_spread_*, n_large_trades,
            large_trade_volume, hidden_absorption_*, net_*_decrement_no_trade_L1,
            quote_update_count.

    Adds (when the inputs are present):
        vwap_eff_spread                 (T1.35) eff_spread_sum / eff_spread_weight
        vwap_eff_spread_buy             (T1.36) eff_spread_buy_sum / eff_spread_buy_weight
        vwap_eff_spread_sell            (T1.36) eff_spread_sell_sum / eff_spread_sell_weight
        eff_spread_asymmetry            (T1.37) buy − sell vwap-eff-spread
        large_trade_volume_share        (T1.23) large_trade_volume / volume
        n_large_trades_log              log(1 + n_large_trades) for Gaussian-like scale
        hidden_absorption_ratio_w{N}    (T1.47/T7.12) rolling sum / rolling sum
        cancel_to_trade_ratio           (T1.43) (bid + ask) cancel decrement / volume
        quote_to_trade_ratio            (T1.24) quote_update_count / trades_count

    Coerces phase_e_bars[ts] dtype to match bars[ts] before asof-join.
    """
    target_dtype = bars.schema[ts_col]
    pe_cols = [c for c in phase_e_bars.columns if c != ts_col]
    pe = (
        phase_e_bars
        .with_columns(pl.col(ts_col).cast(target_dtype))
        .sort(ts_col)
    )
    bars = bars.sort(ts_col).join_asof(pe, on=ts_col, strategy="backward")

    derived: list[pl.Expr] = []

    if {"eff_spread_sum", "eff_spread_weight"}.issubset(pe_cols):
        derived.append(
            (pl.col("eff_spread_sum") / (pl.col("eff_spread_weight") + EPS))
                .alias("vwap_eff_spread")
        )
    if {"eff_spread_buy_sum", "eff_spread_buy_weight"}.issubset(pe_cols):
        derived.append(
            (pl.col("eff_spread_buy_sum") / (pl.col("eff_spread_buy_weight") + EPS))
                .alias("vwap_eff_spread_buy")
        )
    if {"eff_spread_sell_sum", "eff_spread_sell_weight"}.issubset(pe_cols):
        derived.append(
            (pl.col("eff_spread_sell_sum") / (pl.col("eff_spread_sell_weight") + EPS))
                .alias("vwap_eff_spread_sell")
        )
    if {"large_trade_volume"}.issubset(pe_cols) and "volume" in bars.columns:
        derived.append(
            (pl.col("large_trade_volume") / (pl.col("volume") + EPS))
                .alias("large_trade_volume_share")
        )
    if "n_large_trades" in pe_cols:
        derived.append((1 + pl.col("n_large_trades")).log().alias("n_large_trades_log"))
    if {"net_bid_decrement_no_trade_L1", "net_ask_decrement_no_trade_L1"}.issubset(pe_cols) \
            and "volume" in bars.columns:
        derived.append(
            ((pl.col("net_bid_decrement_no_trade_L1") + pl.col("net_ask_decrement_no_trade_L1"))
             / (pl.col("volume") + EPS)).alias("cancel_to_trade_ratio")
        )
    if "quote_update_count" in pe_cols and "trades_count" in bars.columns:
        derived.append(
            (pl.col("quote_update_count") / (pl.col("trades_count") + EPS))
                .alias("quote_to_trade_ratio")
        )

    bars = bars.with_columns(derived)

    # Symmetric eff-spread asymmetry — depends on pass-1 buy/sell vwap cols.
    if "vwap_eff_spread_buy" in bars.columns and "vwap_eff_spread_sell" in bars.columns:
        bars = bars.with_columns(
            (pl.col("vwap_eff_spread_buy") - pl.col("vwap_eff_spread_sell"))
                .alias("eff_spread_asymmetry")
        )

    # Hidden-absorption rolling ratio (T1.47 / T7.12 prerequisite)
    if "hidden_absorption_volume" in pe_cols and "volume" in bars.columns:
        bars = bars.with_columns(
            (pl.col("hidden_absorption_volume").rolling_sum(window_size=hidden_absorption_window)
             / (pl.col("volume").rolling_sum(window_size=hidden_absorption_window) + EPS))
                .alias(f"hidden_absorption_ratio_w{hidden_absorption_window}")
        )
    return bars


# ---------------------------------------------------------------------------
# Convenience: full per-instrument feature pass
# ---------------------------------------------------------------------------

def build_per_instrument_features(
    bars: pl.DataFrame,
    lookback_days_grid: tuple[int, ...] = (30, 60),
    rv_windows: tuple[int, ...] = (20, 60, 120),
    range_vol_windows: tuple[int, ...] = (20, 60),
    vol_surprise_windows: tuple[int, ...] = (20, 60, 120),
    attach_overnight: bool = True,
    attach_patterns: bool = True,
    attach_engines: bool = True,
    attach_smoothed: bool = True,
    attach_cyclic_minute: bool = True,
    fracdiff_d: float = 0.4,
    round_pin_N_grid: tuple[float, ...] = (10.0, 25.0, 50.0),
) -> pl.DataFrame:
    """End-to-end per-instrument feature pipeline.

    Calls in order:
      attach_base_microstructure_features
      attach_session_flags + cyclic minute (from tc_features)
      overnight features (if attach_overnight=True)
      attach_pattern_features (if attach_patterns=True)
      attach_engine_features (if attach_engines=True)
      attach_ts_normalizations on BASE_VALUE_COLS
      attach_ema_smoothed (if attach_smoothed=True)

    Note: L2-deep features (`attach_l2_deep_features`) are NOT called here.
    The caller invokes them separately when working with Phase A+B bars
    that have L1-L10 columns.
    """
    df = attach_base_microstructure_features(
        bars,
        rv_windows=rv_windows,
        range_vol_windows=range_vol_windows,
        vol_surprise_windows=vol_surprise_windows,
    )
    df = tc_features.attach_session_flags(df)
    if attach_cyclic_minute:
        df = tc_features.attach_minute_of_day_cyclic(df)
    if attach_overnight:
        df = overnight.attach_overnight_features(df)
    if attach_patterns:
        df = attach_pattern_features(df)
    if attach_engines:
        df = attach_engine_features(
            df, fracdiff_d=fracdiff_d, round_pin_N_grid=round_pin_N_grid,
        )
    df = attach_ts_normalizations(
        df,
        value_cols=BASE_VALUE_COLS,
        lookback_days_grid=lookback_days_grid,
    )
    if attach_smoothed:
        df = smoothed_features.attach_ema_smoothed(df)
    return df
