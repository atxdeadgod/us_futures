"""Cross-sectional / multi-instrument feature operations.

Operates on a wide frame keyed by `ts`, with columns of the form
`{INSTR}_{value_col}` produced by `build_wide_cross_asset_frame`.

Provides:
    - Universe + asset-class taxonomy constants (TRADING_INSTRUMENTS,
      ASSET_CLASSES, ALL_INSTRUMENTS) — single source of truth for the
      30-contract universe.
    - `build_wide_cross_asset_frame` — joins per-instrument frames on ts.
    - `attach_cross_sectional_ranks` — Gauss-Rank universe + per-asset-class.
    - `attach_cross_asset_composites` — synthetic DXY, rates curve, risk-on/off,
      cross-asset rolling correlations.

All functions here assume `cross_asset_macro` primitives for the actual math
(this module orchestrates them across the wide frame).
"""
from __future__ import annotations

import polars as pl

from . import cross_asset_macro

EPS = 1e-9


# ---------------------------------------------------------------------------
# Universe + asset-class taxonomy
# ---------------------------------------------------------------------------

TRADING_INSTRUMENTS: list[str] = ["ES", "NQ", "RTY", "YM"]

ASSET_CLASSES: dict[str, list[str]] = {
    "EQUITY_INDEX": ["ES", "NQ", "RTY", "YM"],
    "FX":           ["6A", "6B", "6C", "6E", "6J"],
    "ENERGY":       ["BZ", "CL", "HO", "NG", "RB"],
    "METALS":       ["GC", "HG", "PA", "PL", "SI"],
    "RATES":        ["SR3", "TN", "ZB", "ZF", "ZN", "ZT"],
    "AGS":          ["ZC", "ZL", "ZM", "ZS", "ZW"],
}

ALL_INSTRUMENTS: list[str] = [c for cls in ASSET_CLASSES.values() for c in cls]


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
        cols = [f"{i}_{v}" for i in instruments if f"{i}_{v}" in df.columns]
        if len(cols) >= 2:
            df = cross_asset_macro.attach_gauss_rank_cs(
                df, value_cols=cols, out_prefix=f"cs_universe_",
            )
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

    needed_rates = {rates_zt, rates_zf, rates_zn, rates_zb}
    if all(f"{c}_log_return" in df.columns for c in needed_rates):
        df = cross_asset_macro.attach_rates_curve_spreads(
            df,
            zt_logret_col=f"{rates_zt}_log_return",
            zf_logret_col=f"{rates_zf}_log_return",
            zn_logret_col=f"{rates_zn}_log_return",
            zb_logret_col=f"{rates_zb}_log_return",
        )

    macro_refs = {
        "gold": gold,
        "oil":  "CL",
        "ZN":   "ZN",
        "DXY":  None,
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
