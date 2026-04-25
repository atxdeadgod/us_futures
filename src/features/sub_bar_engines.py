"""Sub-bar (5-sec) engine features asof-attached to 15-min bars.

VPIN and Hawkes intensity are statistics derived from the per-trade or per-5-sec
event stream. They cannot be reconstructed from a 15-min bar alone — they need
the finer-resolution series. This module reads 5-sec Phase A bars, computes
the engines per-day, and asof-joins the resulting per-bucket / per-row features
back onto a 15-min bar frame.

Output features (asof-attached to a 15-min bar's `ts`):

VPIN family:
    vpin                              last completed bucket's VPIN value
    vpin_buckets_velocity_w15m        # of bucket completions inside the bar
    vpin_staleness_seconds            seconds since last bucket close at bar close

Hawkes family:
    hawkes_imbalance_hl5              fast (HL=5s)  λ_buy − λ_sell
    hawkes_imbalance_hl60             slow (HL=60s) λ_buy − λ_sell
    hawkes_acceleration               fast − slow (regime-divergence signal)

Bucket sizing for VPIN: defaults to 25_000 contracts/bucket (~1/50th of ES daily
volume per Easley-Lopez de Prado). Override via `vpin_bucket_size`.

Session resets for Hawkes: at the first 5-sec bar of each Globex daily session
(ET hour=18, the new-day open after the 17:00-18:00 daily halt). Sunday 18:00
ET also resets (weekly open). Resets prevent the recursive λ from accumulating
across non-trading windows.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import engines

EPS = 1e-9


def _find_globex_session_resets(
    bars_5s: pl.DataFrame, ts_col: str = "ts",
) -> np.ndarray:
    """Return ts values of the first 5-sec bar of each Globex daily session.

    Bars are labeled at right-edge with closed='left', so a bar with
    ts=18:00:05 ET contains trades [18:00:00, 18:00:05) — the first 5 seconds
    of the new daily session. Use that exact bar as the Hawkes reset point.
    (A bar at ts=18:00:00 holds [17:59:55, 18:00:00) and is still in the prior
    session; not a reset.)
    """
    et = pl.col(ts_col).dt.convert_time_zone("America/New_York")
    return (
        bars_5s.filter(
            (et.dt.hour() == 18) & (et.dt.minute() == 0) & (et.dt.second() == 5)
        )[ts_col].to_numpy()
    )


def compute_vpin_buckets(
    bars_5s: pl.DataFrame,
    bucket_size: int = 25_000,
    ts_col: str = "ts",
    buys_col: str = "buys_qty",
    sells_col: str = "sells_qty",
) -> pl.DataFrame:
    """Per-bucket VPIN time series across the entire 5-sec frame.

    Returns DataFrame [ts, vpin, bucket_volume, n_sub_bars] where ts is the
    bucket-close timestamp. Partial buckets at end-of-input are discarded.
    """
    out = engines.vpin_volume_buckets(
        bars_5s, bucket_size=bucket_size,
        ts_col=ts_col, buys_col=buys_col, sells_col=sells_col,
        keep_partial=False,
    )
    return out.rename({"bucket_close_ts": ts_col})


def compute_hawkes_at_5sec(
    bars_5s: pl.DataFrame,
    hl_seconds_fast: float = 5.0,
    hl_seconds_slow: float = 60.0,
    ts_col: str = "ts",
    buys_col: str = "buys_qty",
    sells_col: str = "sells_qty",
) -> pl.DataFrame:
    """Per-row Hawkes fast/slow imbalance + acceleration at 5-sec resolution.

    Returns DataFrame [ts, hawkes_imbalance_hl{X_fast}, hawkes_imbalance_hl{X_slow},
    hawkes_acceleration]. Resets at each Globex daily session start (~18:00 ET).
    """
    df = bars_5s.sort(ts_col)
    ts_ns = df[ts_col].cast(pl.Int64).to_numpy()
    ts_seconds = ts_ns / 1e9
    buys = df[buys_col].to_numpy().astype(np.float64)
    sells = df[sells_col].to_numpy().astype(np.float64)

    resets = _find_globex_session_resets(df, ts_col=ts_col)
    reset_seconds = (resets.astype(np.int64) / 1e9) if len(resets) > 0 else None

    fast = engines.hawkes_intensity_recursive(
        ts_seconds, buys, sells,
        hl_seconds=hl_seconds_fast,
        session_reset_ts=reset_seconds,
    )
    slow = engines.hawkes_intensity_recursive(
        ts_seconds, buys, sells,
        hl_seconds=hl_seconds_slow,
        session_reset_ts=reset_seconds,
    )

    fast_col = f"hawkes_imbalance_hl{int(hl_seconds_fast)}"
    slow_col = f"hawkes_imbalance_hl{int(hl_seconds_slow)}"
    return df.select(ts_col).with_columns([
        pl.Series(fast_col, fast["imbalance"]),
        pl.Series(slow_col, slow["imbalance"]),
        pl.Series("hawkes_acceleration", fast["imbalance"] - slow["imbalance"]),
    ])


def attach_sub_bar_engine_features(
    bars_15m: pl.DataFrame,
    bars_5s: pl.DataFrame,
    vpin_bucket_size: int = 25_000,
    hawkes_hl_fast: float = 5.0,
    hawkes_hl_slow: float = 60.0,
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Compute VPIN + Hawkes from `bars_5s` and asof-attach to `bars_15m`.

    Adds 6 columns to `bars_15m`:
        vpin, vpin_buckets_velocity_w15m, vpin_staleness_seconds,
        hawkes_imbalance_hl{fast}, hawkes_imbalance_hl{slow}, hawkes_acceleration

    Both the 5-sec source frame and the 15-min target frame must have a `ts`
    column. The function coerces 5-sec ts to match the 15-min frame's ts dtype
    before joining — guards against datetime[ns]/datetime[μs] mismatches.
    """
    target_dtype = bars_15m.schema[ts_col]
    bars_15m = bars_15m.sort(ts_col)
    bars_5s = bars_5s.sort(ts_col)

    # ---- VPIN ----
    vpin_df = compute_vpin_buckets(
        bars_5s, bucket_size=vpin_bucket_size, ts_col=ts_col,
    ).with_columns(pl.col(ts_col).cast(target_dtype)).sort(ts_col)

    # asof-join (backward): vpin value of the last completed bucket ≤ bar close.
    # Carry _vpin_close_ts to compute staleness afterwards.
    vpin_for_join = vpin_df.select([
        ts_col,
        pl.col("vpin"),
        pl.col(ts_col).alias("_vpin_close_ts"),
    ])
    bars_15m = bars_15m.join_asof(
        vpin_for_join, on=ts_col, strategy="backward",
    )
    bars_15m = bars_15m.with_columns(
        ((pl.col(ts_col).cast(pl.Int64) - pl.col("_vpin_close_ts").cast(pl.Int64)) / 1e9)
            .alias("vpin_staleness_seconds")
    ).drop("_vpin_close_ts")

    # vpin_buckets_velocity_w15m: count of bucket closes inside each 15-min window
    bucket_velocity = (
        vpin_df.select(ts_col)
        .group_by_dynamic(ts_col, every="15m", closed="left", label="right")
        .agg(pl.len().alias("vpin_buckets_velocity_w15m"))
        .with_columns(pl.col(ts_col).cast(target_dtype))
    )
    bars_15m = bars_15m.join(
        bucket_velocity, on=ts_col, how="left",
    ).with_columns(
        pl.col("vpin_buckets_velocity_w15m").fill_null(0)
    )

    # ---- Hawkes ----
    hawkes_df = compute_hawkes_at_5sec(
        bars_5s,
        hl_seconds_fast=hawkes_hl_fast, hl_seconds_slow=hawkes_hl_slow,
        ts_col=ts_col,
    ).with_columns(pl.col(ts_col).cast(target_dtype)).sort(ts_col)

    bars_15m = bars_15m.join_asof(
        hawkes_df, on=ts_col, strategy="backward",
    )
    return bars_15m
