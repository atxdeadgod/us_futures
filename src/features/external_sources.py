"""Features derived from non-bar data sources, asof-attached to a target.

Each function takes a target's bar frame and an external-source frame (or
file path), then asof-joins the source onto the target's `ts` and emits
derived feature columns.

Sources covered:
    - VX1/VX2/VX3 futures bars  → vol-regime features (vx mid/zscore/calendar/curvature)
    - SPX (or NDX) options chain GEX profile → distance-to-{zero-gamma, max OI} features

Adding a new external source: write `attach_<source>_for_target` here and
have the upstream orchestrator (`build_single_panel.py`) call it.
"""
from __future__ import annotations

import polars as pl

from . import gex as gex_features
from . import vx as vx_features


# ---------------------------------------------------------------------------
# VX (front-three monthly VIX futures) — vol-regime features
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
    # Coerce ts to a consistent precision before asof-join. VX bars built before
    # the ingest._parse_ts fix have datetime[μs, UTC] (because polars promoted
    # ns→μs when adding a millisecond duration). Newer builds have ns. Force
    # both sides to ns to avoid SchemaError on join_asof.
    target_ts_dtype = target_bars.schema[ts_col]

    def _prep(slot_df: pl.DataFrame, slot: str) -> pl.DataFrame:
        return slot_df.select([
            pl.col(ts_col).cast(target_ts_dtype),
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
# GEX (SPX/SPY options) — distance-to-strike features
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
        daily_gex_paths: list of parquet paths produced by
            scripts/build_gex_features.py; typically one per year.
        es_spx_basis: optional DataFrame [date, basis] where basis = ES_close − SPX_close
            on each date. If None, basis is set to 0 for all dates (coarse but
            adequate for V1; refine with actual basis data in V1.5).
        ts_col, close_col: target bar column names.

    Adds the columns produced by `src/features/gex.attach_gex_features`.
    """
    if not daily_gex_paths:
        return bars
    daily = pl.concat(
        [pl.scan_parquet(p) for p in daily_gex_paths],
        how="diagonal_relaxed",
    ).collect()
    if es_spx_basis is None:
        es_spx_basis = daily.select("date").with_columns(pl.lit(0.0).alias("basis"))
    return gex_features.attach_gex_features(
        bars, daily, es_spx_basis, ts_col=ts_col, close_col=close_col,
    )
