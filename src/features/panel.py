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
from . import overnight
from . import tc_features

EPS = 1e-9


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
# layout downstream.
BASE_VALUE_COLS = [
    "log_return",
    "abs_log_return",
    "log_volume",
    "ofi",
    "aggressor_ratio",
    "cvd_change",
    "spread_to_mid_bps",
    "realized_vol_w20",
    "range_vol_parkinson_w20",
    "vol_surprise_w60",
]


# ---------------------------------------------------------------------------
# Step 1 — per-instrument base microstructure features
# ---------------------------------------------------------------------------

def attach_base_microstructure_features(
    bars: pl.DataFrame,
    rv_window: int = 20,
    range_vol_window: int = 20,
    vol_surprise_window: int = 60,
) -> pl.DataFrame:
    """Compute base microstructure features from Phase A bars.

    Phase A bars must include: ts, open, high, low, close, volume,
    buys_qty, sells_qty, mid_close, spread_abs_close, cvd_globex.

    Output columns appended (one per BASE_VALUE_COLS entry):
        log_return                 log(close / close.shift(1))
        abs_log_return             |log_return|
        log_volume                 log(1 + volume)
        ofi                        (buys − sells) / (buys + sells + EPS)
        aggressor_ratio            buys / (buys + sells + EPS)
        cvd_change                 cvd_globex − cvd_globex.shift(1)
        spread_to_mid_bps          spread_abs_close / mid_close × 10_000
        realized_vol_w20           rolling std of log_return (window=rv_window)
        range_vol_parkinson_w20    Parkinson range vol (window=range_vol_window)
        vol_surprise_w60           log(volume) − rolling_mean(log(volume), window=vol_surprise_window)

    Returns the original bars DataFrame plus these feature columns.
    """
    df = bars.with_columns([
        # log_return + abs_log_return
        bar_features.log_return(pl.col("close")).alias("log_return"),
    ])
    df = df.with_columns([
        pl.col("log_return").abs().alias("abs_log_return"),
        bar_features.log_volume(pl.col("volume")).alias("log_volume"),
    ])
    df = df.with_columns([
        # OFI from aggressor split (buys/sells)
        ((pl.col("buys_qty") - pl.col("sells_qty"))
         / (pl.col("buys_qty") + pl.col("sells_qty") + EPS)).alias("ofi"),
        (pl.col("buys_qty") / (pl.col("buys_qty") + pl.col("sells_qty") + EPS))
            .alias("aggressor_ratio"),
        # CVD change (per-bar increment of cumulative volume delta)
        (pl.col("cvd_globex") - pl.col("cvd_globex").shift(1)).alias("cvd_change"),
        # Spread (in bps of mid)
        (pl.col("spread_abs_close") / (pl.col("mid_close") + EPS) * 10_000)
            .alias("spread_to_mid_bps"),
        # Realized vol (rolling std of log_return)
        bar_features.realized_volatility(pl.col("log_return"), window=rv_window, method="std")
            .alias(f"realized_vol_w{rv_window}"),
        # Parkinson range vol
        bar_features.range_vol_parkinson(pl.col("high"), pl.col("low"), window=range_vol_window)
            .alias(f"range_vol_parkinson_w{range_vol_window}"),
    ])
    df = df.with_columns([
        # Volume surprise (log_volume minus its rolling mean)
        (pl.col("log_volume")
         - pl.col("log_volume").rolling_mean(window_size=vol_surprise_window))
            .alias(f"vol_surprise_w{vol_surprise_window}"),
    ])
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
# Convenience: full per-instrument feature pass (Steps 1+2)
# ---------------------------------------------------------------------------

def build_per_instrument_features(
    bars: pl.DataFrame,
    lookback_days_grid: tuple[int, ...] = (30, 60),
    rv_window: int = 20,
    range_vol_window: int = 20,
    vol_surprise_window: int = 60,
) -> pl.DataFrame:
    """End-to-end per-instrument feature pipeline (steps 1 + 2)."""
    df = attach_base_microstructure_features(
        bars,
        rv_window=rv_window,
        range_vol_window=range_vol_window,
        vol_surprise_window=vol_surprise_window,
    )
    df = attach_ts_normalizations(
        df,
        value_cols=BASE_VALUE_COLS,
        lookback_days_grid=lookback_days_grid,
    )
    return df
