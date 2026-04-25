"""Cross-asset feature panel orchestrator.

Pipeline (per target trading instrument):

    Phase A bars (30 contracts) → per-contract base microstructure features
    → time-series normalizations (TC z, MAD z, multiple windows)
    → wide cross-asset join (one row per ts, columns prefixed by instrument)
    → cross-sectional Gauss-Rank features (universe + within asset class)
    → cross-asset composites (synthetic DXY, rates curve, risk-on/off)
    → target instrument's L2 features (Phase A+B for ES/NQ/RTY/YM)
    → GEX features (SPX → ES; later NDX → NQ etc.)
    → VX features (when sync completes)
    → labels (triple-barrier with V1 locked params)
    → final panel parquet

This module focuses on STEPS 1-2 (per-instrument base features +
normalizations). Steps 3-7 are in subsequent functions/modules added as
the bar chain delivers data.
"""
from __future__ import annotations

import polars as pl

from . import bar as bar_features
from . import bar_agg
from . import cross_asset_macro
from . import deep_ofi as deep_ofi_features
from . import engines as engine_features
from . import gex as gex_features
from . import l1 as l1_features
from . import l2 as l2_features
from . import overnight
from . import patterns as pattern_features
from . import smoothed as smoothed_features
from . import tc_features
from . import vx as vx_features
from ..labels.triple_barrier import triple_barrier_labels

EPS = 1e-9


# ---------------------------------------------------------------------------
# V1 locked label params per trading instrument (from LABELING_V1_SUMMARY.md)
# ---------------------------------------------------------------------------

V1_LABEL_PARAMS: dict[str, dict] = {
    "ES":  {"k_up": 1.25, "k_dn": 1.25, "T": 8, "lookback_days": 150},
    "NQ":  {"k_up": 1.25, "k_dn": 1.25, "T": 8, "lookback_days": 180},
    "RTY": {"k_up": 1.00, "k_dn": 1.00, "T": 4, "lookback_days": 180},
    "YM":  {"k_up": 1.25, "k_dn": 1.25, "T": 8, "lookback_days": 150},
}

# Round-trip cost (instrument points; spread + commission + slippage estimates)
V1_COST_PTS: dict[str, float] = {"ES": 0.50, "NQ": 1.50, "RTY": 0.30, "YM": 3.00}


# ---------------------------------------------------------------------------
# Universe + asset-class taxonomy (must match TODO.md / bar build)
# ---------------------------------------------------------------------------

TRADING_INSTRUMENTS = ["ES", "NQ", "RTY", "YM"]

ASSET_CLASSES: dict[str, list[str]] = {
    "EQUITY_INDEX": ["ES", "NQ", "RTY", "YM"],
    "FX":           ["6A", "6B", "6C", "6E", "6J"],
    "ENERGY":       ["BZ", "CL", "HO", "NG", "RB"],
    "METALS":       ["GC", "HG", "PA", "PL", "SI"],
    "RATES":        ["SR3", "TN", "ZB", "ZF", "ZN", "ZT"],
    "AGS":          ["ZC", "ZL", "ZM", "ZS", "ZW"],
}

ALL_INSTRUMENTS: list[str] = [c for cls in ASSET_CLASSES.values() for c in cls]

# Base values computed per instrument that get TC+MAD z-scored AND
# cross-sectionally Gauss-Ranked downstream. Order matters for column
# layout downstream. Extended set covers higher moments, asymmetric vol,
# multiple windows, autocorrelation, illiquidity — see expand block in
# attach_base_microstructure_features for the full list of computed cols.
BASE_VALUE_COLS = [
    # Core returns & flow
    "log_return", "abs_log_return", "log_volume",
    "ofi", "aggressor_ratio", "cvd_change", "net_aggressor_volume",
    "average_trade_size", "trades_count_log",
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
    # ---- Pass 1: pure column-wise transforms ----
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
    df = bars.with_columns(pass1).with_columns([
        pl.col("log_return").abs().alias("abs_log_return"),
    ])

    # ---- Pass 2: rolling stats on derived series ----
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

    # ---- Pass 3: derived from pass-2 outputs ----
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

    # vwap_deviation: rolling typical-price-weighted-volume mean over the day
    # (per ts, daily reset). Use ET trading-date as the partition.
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

    For tree models, both are useful: TC z (mean/std) is sensitive to outliers
    affecting std; MAD z stays informative under tail events. ILP picks per-feature
    which carries more signal.
    """
    df = bars
    for value_col in value_cols:
        if value_col not in df.columns:
            continue  # silently skip — caller may pass a feature set that doesn't apply
        for lb in lookback_days_grid:
            # TC z-score
            df = tc_features.attach_tc_zscore(
                df, value_col=value_col, lookback_days=lb,
                bar_minutes=bar_minutes, partition_minutes=partition_minutes,
                ts_col=ts_col,
                out_col=f"{value_col}_tc_z_w{lb}",
            )
            # MAD z-score
            df = cross_asset_macro.attach_mad_zscore(
                df, value_col=value_col, lookback_days=lb,
                bar_minutes=bar_minutes, partition_minutes=partition_minutes,
                ts_col=ts_col,
                out_col=f"{value_col}_tc_madz_w{lb}",
            )
    return df


# ---------------------------------------------------------------------------
# Step 3 — wide cross-asset join
# ---------------------------------------------------------------------------

def build_wide_cross_asset_frame(
    per_instrument_frames: dict[str, pl.DataFrame],
    base_value_cols: list[str],
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Join per-instrument feature frames on ts → wide cross-asset frame.

    For each instrument, retain only `ts` + `base_value_cols`; rename feature
    columns to `{INSTR}_{value_col}` so the wide frame has one column per
    (instrument, base value) at each ts.

    Outer-joins on ts so a missing day for one contract doesn't drop other
    contracts' data; nulls propagate.
    """
    frames = []
    for instr, df in per_instrument_frames.items():
        avail = [c for c in base_value_cols if c in df.columns]
        if not avail:
            continue
        renamed = {c: f"{instr}_{c}" for c in avail}
        sub = df.select([ts_col] + avail).rename(renamed)
        frames.append(sub)

    if not frames:
        raise ValueError("No per-instrument frames had any of the requested base_value_cols")

    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, on=ts_col, how="full", coalesce=True)
    return wide.sort(ts_col)


# ---------------------------------------------------------------------------
# Step 4 — cross-sectional Gauss-Rank
# ---------------------------------------------------------------------------

def attach_cross_sectional_ranks(
    wide: pl.DataFrame,
    base_value_cols: list[str],
    instruments: list[str],
    asset_classes: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """For each base value, attach Gauss-Rank across the universe AND within asset class.

    For base value `v`:
        wide has columns {instr}_{v} for each instrument.
        Output adds:
          gauss_rank_universe_{instr}_{v}      — rank vs all 30 contracts
          gauss_rank_class_{instr}_{v}         — rank vs same-class peers (if asset_classes given)
    """
    df = wide
    for v in base_value_cols:
        # Universe rank
        cols = [f"{i}_{v}" for i in instruments if f"{i}_{v}" in df.columns]
        if len(cols) >= 2:
            df = cross_asset_macro.attach_gauss_rank_cs(
                df, value_cols=cols, out_prefix=f"cs_universe_",
            )
        # Within-asset-class rank
        if asset_classes is not None:
            for class_name, class_instrs in asset_classes.items():
                cls_cols = [f"{i}_{v}" for i in class_instrs if f"{i}_{v}" in df.columns]
                if len(cls_cols) >= 2:
                    df = cross_asset_macro.attach_gauss_rank_cs(
                        df, value_cols=cls_cols,
                        out_prefix=f"cs_class_{class_name}_",
                    )
    return df


# ---------------------------------------------------------------------------
# Step 5 — cross-asset composites
# ---------------------------------------------------------------------------

def attach_cross_asset_composites(
    wide: pl.DataFrame,
    fx_eur: str = "6E", fx_jpy: str = "6J", fx_gbp: str = "6B", fx_cad: str = "6C",
    rates_zt: str = "ZT", rates_zf: str = "ZF", rates_zn: str = "ZN", rates_zb: str = "ZB",
    gold: str = "GC",
    rolling_corr_window: int = 60,
) -> pl.DataFrame:
    """Synthetic DXY, rates curve, risk-on/off composite, cross-asset rolling corrs.

    Requires log_return columns to exist for each referenced instrument.
    """
    df = wide

    # Synthetic DXY (from FX log returns)
    needed_fx = {fx_eur, fx_jpy, fx_gbp, fx_cad}
    if all(f"{c}_log_return" in df.columns for c in needed_fx):
        df = cross_asset_macro.attach_synthetic_dxy_logret(
            df,
            eur_logret_col=f"{fx_eur}_log_return",
            jpy_logret_col=f"{fx_jpy}_log_return",
            gbp_logret_col=f"{fx_gbp}_log_return",
            cad_logret_col=f"{fx_cad}_log_return",
            out_col="synthetic_dxy_logret",
        )

    # Rates curve spreads
    needed_rates = {rates_zt, rates_zf, rates_zn, rates_zb}
    if all(f"{c}_log_return" in df.columns for c in needed_rates):
        df = cross_asset_macro.attach_rates_curve_spreads(
            df,
            zt_logret_col=f"{rates_zt}_log_return",
            zf_logret_col=f"{rates_zf}_log_return",
            zn_logret_col=f"{rates_zn}_log_return",
            zb_logret_col=f"{rates_zb}_log_return",
        )

    # Cross-asset rolling correlations: target trading instruments × macro families
    macro_refs = {
        "gold": gold,
        "oil":  "CL",
        "ZN":   "ZN",
        "DXY":  None,   # synthetic, attach below if available
    }
    for target in TRADING_INSTRUMENTS:
        target_lr = f"{target}_log_return"
        if target_lr not in df.columns:
            continue
        for label, ref in macro_refs.items():
            if ref is None:
                ref_lr = "synthetic_dxy_logret" if "synthetic_dxy_logret" in df.columns else None
            else:
                ref_lr = f"{ref}_log_return" if f"{ref}_log_return" in df.columns else None
            if ref_lr is None or ref_lr == target_lr:
                continue
            df = cross_asset_macro.attach_rolling_correlation(
                df, col_a=target_lr, col_b=ref_lr,
                window=rolling_corr_window,
                out_col=f"corr_{target}_vs_{label}_w{rolling_corr_window}",
            )
    return df


# ---------------------------------------------------------------------------
# Step 5b — L2-deep features (only on Phase A+B bars for trading instruments)
# ---------------------------------------------------------------------------

def attach_l2_deep_features(
    bars: pl.DataFrame,
    depth: int = 10,
    spread_z_window: int = 60,
) -> pl.DataFrame:
    """Attach L2-derived features to bars that have L1-L10 columns.

    Requires bid_px_L{1..depth}, ask_px_L{1..depth}, bid_sz_L{1..depth},
    ask_sz_L{1..depth} columns (output of `src/data/depth_snap.attach_book_snapshot`,
    Track B Phase A+B bars).

    Adds:
        cum_imbalance_d{depth}        cumulative book imbalance over levels 1..depth
        dw_imbalance_d{depth}         distance-weighted book imbalance
        depth_weighted_spread_d{depth}
        liquidity_adjusted_spread_d{depth}
        spread_acceleration
        hhi_d{depth}                  Herfindahl-Hirschman concentration of depth
        deep_ofi_d{depth}_decay0      multi-level OFI, no decay
        deep_ofi_d{depth}_decay03     multi-level OFI, exp decay = 0.3
        spread_zscore_w{spread_z_window}  rolling z of L1 spread (calendar window)
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
    # Per-level features (volume / OFI / spread / order-count / order-size)
    for k in range(1, depth + 1):
        aggs.append(l2_features.volume_imbalance_at(k=k).alias(f"volume_imbalance_L{k}"))
        aggs.append(l2_features.basic_spread_at(k=k).alias(f"basic_spread_L{k}"))
        aggs.append(deep_ofi_features.ofi_at_level(k=k).alias(f"ofi_at_L{k}"))
        if k <= 5:  # order count/size only for top 5 (signal saturates beyond)
            aggs.append(l2_features.order_count_imbalance_at(k=k).alias(f"order_count_imbalance_L{k}"))
            aggs.append(l2_features.order_size_imbalance_at(k=k).alias(f"order_size_imbalance_L{k}"))
    return bars.with_columns(aggs)


# ---------------------------------------------------------------------------
# Step 5d — pattern features (T7.* — absorption / breakouts / divergence)
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

    Adds:
        atr_proxy
        absorption_score_w20
        breakout_magnitude_up / down
        breakout_reversal_up / down
        post_breakout_flow_reversal_up / down
        spike_and_fade_volume
        imbalance_persistence_runlength_w30
        cvd_price_divergence_up / down
        range_compression_ratio_w20
        volume_at_price_concentration  (via large-trade proxy if present)
    """
    # Build an ATR proxy from Parkinson range vol × close (in price units)
    # if it's not already there. Range vol is in log units → multiply by close.
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
        # Absorption: |dollar-flow| / (vol × notional), proxy aggressor_dollar = (buys-sells)*close
        pattern_features.absorption_score(
            ((pl.col("buys_qty") - pl.col("sells_qty")) * pl.col("close")),
            pl.col(f"realized_vol_w20") if "realized_vol_w20" in bars.columns
                else pl.col("log_return").rolling_std(window_size=20),
            (pl.col("close") * pl.col("volume")),
            window=20,
        ).alias("absorption_score_w20"),
    ]
    df = bars.with_columns(aggs)

    # Post-breakout flow reversal: sums aggressor sign over next k bars after a flag.
    # Use OFI sign as per-bar aggressor sign proxy.
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
# Step 5e — engine-level features (fracdiff, round_number_pin)
# ---------------------------------------------------------------------------

def attach_engine_features(
    bars: pl.DataFrame,
    fracdiff_d: float = 0.4,
    fracdiff_tau: float = 1e-5,
    round_pin_N_grid: tuple[float, ...] = (10.0, 25.0, 50.0),
    close_col: str = "close",
) -> pl.DataFrame:
    """Attach engine-level features: fixed-d FFD on log(close), round-number-pin distance.

    Args:
        fracdiff_d: order parameter for FFD; 0.4 is a common stationarity-preserving
            choice for log-price series. Use fracdiff_auto_d offline to pick a
            data-driven d, then hard-code it here.
        round_pin_N_grid: list of N values to compute pin distance for (e.g. for ES,
            (10, 25, 50) gives distance to nearest 10-pt, 25-pt, 50-pt strike).
    """
    log_close = bars.select(pl.col(close_col).log().alias("_lc"))["_lc"]
    fd = engine_features.fracdiff_series(log_close, d=fracdiff_d, tau=fracdiff_tau)
    df = bars.with_columns(pl.Series(f"fracdiff_logclose_d{fracdiff_d}", fd))
    for N in round_pin_N_grid:
        rn = engine_features.round_number_pin_distance(bars[close_col], N=float(N))
        df = df.with_columns(pl.Series(f"round_pin_distance_N{int(N)}", rn))
    return df


# ---------------------------------------------------------------------------
# Step 5f — VX features (attached onto target via asof-join)
# ---------------------------------------------------------------------------

def attach_vx_features(
    target_bars: pl.DataFrame,
    vx1_bars: pl.DataFrame,
    vx2_bars: pl.DataFrame | None = None,
    vx3_bars: pl.DataFrame | None = None,
    ts_col: str = "ts",
    zscore_window: int = 20,
    spread_z_window: int = 60,
) -> pl.DataFrame:
    """Attach VX1/VX2/VX3 mid/spread to target bars and compute vx.py expressions.

    Args:
        target_bars: per-bar target (e.g., ES 15m bars with ts column)
        vx1_bars/vx2_bars/vx3_bars: stitched-across-days Phase A bars for each
            slot. Each must have `ts`, `bid_close`, `ask_close`, `mid_close`,
            `spread_abs_close` columns.

    Adds (when slot is provided):
        vx1_mid, vx2_mid, vx3_mid
        vx1_zscore_w{zscore_window}
        vx_calendar_spread, vx_calendar_ratio
        vx_spread_zscore_w{spread_z_window}
        vx_term_curvature                 (only if all three slots given)
    """
    def _prep(slot_df: pl.DataFrame, slot: str) -> pl.DataFrame:
        return slot_df.select([
            ts_col,
            pl.col("bid_close").alias(f"{slot}_bid_px_L1"),
            pl.col("ask_close").alias(f"{slot}_ask_px_L1"),
            pl.col("mid_close").alias(f"{slot}_mid_close"),
            pl.col("spread_abs_close").alias(f"{slot}_spread_abs_close"),
        ]).sort(ts_col)

    df = target_bars.sort(ts_col)
    df = df.join_asof(_prep(vx1_bars, "VX1"), on=ts_col, strategy="backward")
    if vx2_bars is not None:
        df = df.join_asof(_prep(vx2_bars, "VX2"), on=ts_col, strategy="backward")
    if vx3_bars is not None:
        df = df.join_asof(_prep(vx3_bars, "VX3"), on=ts_col, strategy="backward")

    aggs = [
        vx_features.vx_mid("VX1").alias("vx1_mid"),
        vx_features.vx_zscore(vx_features.vx_mid("VX1"), window=zscore_window)
            .alias(f"vx1_zscore_w{zscore_window}"),
    ]
    if vx2_bars is not None:
        aggs.extend([
            vx_features.vx_mid("VX2").alias("vx2_mid"),
            vx_features.vx_calendar_spread("VX1", "VX2").alias("vx_calendar_spread"),
            vx_features.vx_calendar_ratio("VX1", "VX2").alias("vx_calendar_ratio"),
        ])
    if vx3_bars is not None:
        aggs.extend([
            vx_features.vx_mid("VX3").alias("vx3_mid"),
            vx_features.vx_term_curvature("VX1", "VX2", "VX3").alias("vx_term_curvature"),
        ])
    aggs.append(
        vx_features.vx_spread_zscore("VX1", depth=1, window=spread_z_window)
            .alias(f"vx_spread_zscore_w{spread_z_window}")
    )
    return df.with_columns(aggs)


# ---------------------------------------------------------------------------
# Step 5c — GEX features (target = ES in V1; later wire NDX→NQ etc.)
# ---------------------------------------------------------------------------

def attach_gex_for_target(
    bars: pl.DataFrame,
    daily_gex_paths: list[str],
    es_spx_basis: pl.DataFrame | None = None,
    ts_col: str = "ts",
    close_col: str = "close",
) -> pl.DataFrame:
    """Attach SPX-derived GEX features to ES bars.

    Args:
        bars: ES bars with ts + close
        daily_gex_paths: list of parquet paths produced by Track C
            (build_gex_features.py); typically one per year.
        es_spx_basis: optional DataFrame [date, basis] where basis = ES_close − SPX_close
            on each date. If None, basis is set to 0 for all dates (coarse but
            adequate for V1; refine with actual basis data in V1.5).
        ts_col, close_col: target bar column names.

    Adds the columns produced by `src/features/gex.attach_gex_features`.
    """
    if not daily_gex_paths:
        return bars
    daily = pl.concat([pl.scan_parquet(p) for p in daily_gex_paths],
                       how="vertical_relaxed").collect()
    if es_spx_basis is None:
        es_spx_basis = daily.select("date").with_columns(pl.lit(0.0).alias("basis"))
    return gex_features.attach_gex_features(
        bars, daily, es_spx_basis, ts_col=ts_col, close_col=close_col,
    )


# ---------------------------------------------------------------------------
# Step 6 — assemble final per-target panel (cross-asset + L2 + GEX + labels)
# ---------------------------------------------------------------------------

def assemble_target_panel(
    target: str,
    target_bars_with_features: pl.DataFrame,
    wide_cross_asset: pl.DataFrame | None = None,
    label_params: dict | None = None,
    cost_pts: float | None = None,
    halt_mode: str = "truncate",
    min_effective_T: int = 5,
    partition_minutes: int = 15,
    bar_minutes: int = 15,
    drop_invalid: bool = True,
) -> pl.DataFrame:
    """Final per-target feature panel.

    Workflow:
      1. Start with target's own bars-with-features (Step 1+2 already applied,
         optionally with overnight + L2 features attached upstream).
      2. Left-join the wide cross-asset frame (Steps 3-5 output) on `ts`,
         dropping the wide frame's `{target}_*` columns (already in target's bars).
      3. Apply triple_barrier_labels with V1 locked params for `target`.
      4. Filter to valid rows (atr + realized_ret finite) — drops warmup +
         halt-truncated bars where effective T < min_effective_T.

    Args:
        target: trading instrument key (must be in V1_LABEL_PARAMS)
        target_bars_with_features: target's Phase A (or A+B) bars with base
            features + normalizations + (optionally) overnight + L2 features
        wide_cross_asset: output of Steps 3-5; if None, only target's own
            features go into the panel
        label_params: V1 label params for target. If None, uses V1_LABEL_PARAMS[target].
        cost_pts: round-trip cost. If None, uses V1_COST_PTS[target].
        halt_mode, min_effective_T, partition_minutes: V1 architecture defaults
            from LABELING_V1_SUMMARY.md (truncate, 5, 15)
        drop_invalid: filter out rows where atr or realized_ret is non-finite
            (warmup, halt-truncated, etc.). Default True.

    Returns:
        Per-bar feature panel with `label`, `realized_ret`, `realized_ret_pts`,
        `hit_offset`, `halt_truncated`, `atr` columns appended.
    """
    if label_params is None:
        if target not in V1_LABEL_PARAMS:
            raise ValueError(f"No V1_LABEL_PARAMS for target={target!r}; provide label_params explicitly")
        label_params = V1_LABEL_PARAMS[target]

    df = target_bars_with_features

    # Drop wide frame's {target}_* cols since target's bars are authoritative for those
    if wide_cross_asset is not None:
        target_prefix_cols = [c for c in wide_cross_asset.columns if c.startswith(f"{target}_")]
        wide_to_join = wide_cross_asset.drop(target_prefix_cols)
        # Avoid duplicate columns post-join (besides ts)
        already = set(df.columns)
        wide_unique_cols = ["ts"] + [c for c in wide_to_join.columns if c != "ts" and c not in already]
        wide_subset = wide_to_join.select(wide_unique_cols)
        df = df.join(wide_subset, on="ts", how="left")

    # Apply labels (TC-ATR + truncate min=5, V1 architecture)
    labeled = triple_barrier_labels(
        df,
        k_up=label_params["k_up"],
        k_dn=label_params["k_dn"],
        T=label_params["T"],
        atr_window=label_params.get("lookback_days", 60),  # TC-ATR carries lookback in this slot
        atr_mode="time_conditional",
        lookback_days=label_params["lookback_days"],
        bar_minutes=bar_minutes,
        partition_minutes=partition_minutes,
        halt_aware=True,
        halt_mode=halt_mode,
        min_effective_T=min_effective_T,
    )

    if drop_invalid:
        labeled = labeled.filter(
            pl.col("atr").is_finite() & pl.col("realized_ret").is_finite()
        )

    return labeled


# ---------------------------------------------------------------------------
# Convenience: full per-instrument feature pass (Steps 1+2)
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
      step 1  attach_base_microstructure_features
      session flags (force consistent hour_et / is_rth / is_eu / is_asia / is_eth
                     using V1 convention; overrides any pre-existing flags)
      overnight features (if attach_overnight=True): per-trading-day overnight
                     gap, realized vol, volume, n_bars — broadcast to every bar
                     in the day. Lets RTH bars use overnight context as features.
      step 2  attach_ts_normalizations on BASE_VALUE_COLS

    Note: overnight columns are NOT in BASE_VALUE_COLS, so they don't get
    TC/MAD z-scored or cross-sectionally ranked in V1. They flow through to
    the final panel as raw daily values per bar. Per-asset overnight ranks
    can be added in V1.5 by extending the wide-join + CS-ranks pipeline.
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
