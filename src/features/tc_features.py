"""Time-conditional (same-bar-of-day) feature variants.

The dual of TC-ATR: instead of the calendar-window z-score / rolling-baseline,
we partition by ET bar-of-day so the "typical" reference is what *this hour*
historically does, not what the past N bars (regardless of hour) did.

Why this matters: the per-hour realized variance / volume / spread / OFI
distributions are wildly different across the 23h CME session. A calendar
z-score for, say, volume_surprise at 22:00 ET will register as "low" because
the reference window is dominated by RTH activity. A TC z-score asks "is this
22:00 ET bar's volume HIGH or LOW relative to typical 22:00 ET bars?" — which
is the operationally relevant question.

This module mirrors the V1 labeling architecture (`triple_barrier.py`):
- partition by 15-min bar-of-day (in US/Eastern, DST-aware)
- rolling mean/std over the last `lookback_days` same-bar-of-day samples
- materialize bar_of_day as an explicit column (polars-version-portable)

Functions (all return a DataFrame with the new column appended):

    attach_tc_zscore(bars, value_col, lookback_days, ...)  → generic primitive
    attach_volume_surprise_tc(bars, ...)
    attach_ofi_zscore_tc(bars, ofi_col, ...)
    attach_spread_zscore_tc(bars, ...)
    attach_realized_vol_zscore_tc(bars, ...)
    attach_session_flags(bars, ts_col)   → is_asia, is_eu, is_rth, is_ext
    attach_minute_of_day_cyclic(bars, ts_col)  → sin/cos encoding
"""
from __future__ import annotations

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# Generic TC z-score primitive
# ---------------------------------------------------------------------------

def attach_tc_zscore(
    bars: pl.DataFrame,
    value_col: str,
    lookback_days: int = 30,
    bar_minutes: int = 15,
    partition_minutes: int | None = None,
    ts_col: str = "ts",
    out_col: str | None = None,
) -> pl.DataFrame:
    """Attach `(value − rolling_mean_tc) / rolling_std_tc` for `value_col`.

    The rolling stats partition by bar-of-day in US/Eastern and use the most
    recent `lookback_days` same-bar-of-day observations. Mirrors the TC-ATR
    construction in `src/labels/triple_barrier.py`.

    Args:
        value_col: name of the column to z-score
        lookback_days: window length per partition (in days, since each
            partition has 1 sample per day)
        bar_minutes: bar duration in minutes (15 for 15-min bars)
        partition_minutes: granularity at which to partition (default = bar_minutes)
        out_col: output column name. Defaults to f"{value_col}_tc_z".

    Returns the original frame plus the TC z-score column.
    """
    if partition_minutes is None:
        partition_minutes = bar_minutes
    if 60 % partition_minutes != 0:
        raise ValueError(f"partition_minutes={partition_minutes} must divide 60")
    if out_col is None:
        out_col = f"{value_col}_tc_z"

    parts_per_hour = 60 // partition_minutes
    et = pl.col(ts_col).dt.convert_time_zone("US/Eastern")
    df = bars.with_columns(
        (et.dt.hour() * parts_per_hour + et.dt.minute() // partition_minutes).alias("_bod_tc")
    )
    df = df.with_columns([
        pl.col(value_col).rolling_mean(window_size=lookback_days).over("_bod_tc").alias("_tc_mean"),
        pl.col(value_col).rolling_std(window_size=lookback_days).over("_bod_tc").alias("_tc_std"),
    ])
    df = df.with_columns(
        ((pl.col(value_col) - pl.col("_tc_mean")) / (pl.col("_tc_std") + EPS)).alias(out_col)
    )
    return df.drop(["_bod_tc", "_tc_mean", "_tc_std"])


# ---------------------------------------------------------------------------
# Specific feature wrappers
# ---------------------------------------------------------------------------

def attach_volume_surprise_tc(
    bars: pl.DataFrame,
    volume_col: str = "volume",
    lookback_days: int = 30,
    bar_minutes: int = 15,
    partition_minutes: int | None = None,
    ts_col: str = "ts",
    out_col: str = "volume_surprise_tc",
) -> pl.DataFrame:
    """Volume z-scored against same-bar-of-day baseline.

    Replaces the calendar-window `volume_surprise` for off-hours sanity. At
    22:00 ET, calendar volume_surprise is dominated by RTH-bar values in the
    lookback so the 22:00 bar always looks "low"; TC version asks whether
    THIS 22:00 bar is high relative to typical 22:00 ET bars.
    """
    return attach_tc_zscore(
        bars, volume_col, lookback_days=lookback_days, bar_minutes=bar_minutes,
        partition_minutes=partition_minutes, ts_col=ts_col, out_col=out_col,
    )


def attach_ofi_zscore_tc(
    bars: pl.DataFrame,
    ofi_col: str,
    lookback_days: int = 30,
    bar_minutes: int = 15,
    partition_minutes: int | None = None,
    ts_col: str = "ts",
    out_col: str = "ofi_tc_z",
) -> pl.DataFrame:
    """OFI z-scored against same-bar-of-day baseline.

    OFI magnitude is volume-proportional, so calendar normalization gives
    degenerate values at off-hours. TC normalization asks whether THIS bar's
    OFI is large relative to typical-this-hour OFI.
    """
    return attach_tc_zscore(
        bars, ofi_col, lookback_days=lookback_days, bar_minutes=bar_minutes,
        partition_minutes=partition_minutes, ts_col=ts_col, out_col=out_col,
    )


def attach_spread_zscore_tc(
    bars: pl.DataFrame,
    spread_col: str = "spread_abs_close",
    lookback_days: int = 30,
    bar_minutes: int = 15,
    partition_minutes: int | None = None,
    ts_col: str = "ts",
    out_col: str = "spread_tc_z",
) -> pl.DataFrame:
    """Spread z-scored against same-bar-of-day baseline.

    Spreads structurally widen overnight and tighten in RTH. Calendar z-score
    says "spread is +3σ overnight" every overnight bar (degenerate). TC
    z-score correctly identifies abnormally wide spreads RELATIVE TO THE HOUR.
    """
    return attach_tc_zscore(
        bars, spread_col, lookback_days=lookback_days, bar_minutes=bar_minutes,
        partition_minutes=partition_minutes, ts_col=ts_col, out_col=out_col,
    )


def attach_realized_vol_zscore_tc(
    bars: pl.DataFrame,
    rvol_col: str,
    lookback_days: int = 30,
    bar_minutes: int = 15,
    partition_minutes: int | None = None,
    ts_col: str = "ts",
    out_col: str = "rvol_tc_z",
) -> pl.DataFrame:
    """Realized volatility z-scored against same-bar-of-day baseline."""
    return attach_tc_zscore(
        bars, rvol_col, lookback_days=lookback_days, bar_minutes=bar_minutes,
        partition_minutes=partition_minutes, ts_col=ts_col, out_col=out_col,
    )


# ---------------------------------------------------------------------------
# Time-of-day flags + cyclic encodings
# ---------------------------------------------------------------------------

# Session boundaries in ET (US/Eastern, DST-aware via convert_time_zone).
# These match the labeling-side definitions in triple_barrier.py.
def _session_for_hour(hour_et: int) -> str:
    if 18 <= hour_et or hour_et < 3:    return "ASIA"
    if 3 <= hour_et < 9:                return "EU"
    if 9 <= hour_et < 16:               return "RTH"
    if 16 <= hour_et < 18:              return "ETH"
    return "UNKNOWN"


def attach_session_flags(
    bars: pl.DataFrame,
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Attach is_asia, is_eu, is_rth, is_eth boolean flags + hour_et int.

    All boundaries in US/Eastern (DST-aware). Lets the model learn session-
    conditional patterns (e.g., "RTH OFI predicts differently than EU OFI").
    """
    et = pl.col(ts_col).dt.convert_time_zone("US/Eastern")
    h = et.dt.hour()
    return bars.with_columns([
        h.alias("hour_et"),
        ((h >= 18) | (h < 3)).cast(pl.Int8).alias("is_asia"),
        ((h >= 3) & (h < 9)).cast(pl.Int8).alias("is_eu"),
        ((h >= 9) & (h < 16)).cast(pl.Int8).alias("is_rth"),
        ((h >= 16) & (h < 18)).cast(pl.Int8).alias("is_eth"),
    ])


def attach_minute_of_day_cyclic(
    bars: pl.DataFrame,
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Attach sin/cos cyclic encoding of minute-of-day (in ET).

    Lets the model use time-of-day as a smooth feature without the
    discontinuity at midnight that integer minute_of_day would create.
    Period = 24h = 1440 minutes.
    """
    et = pl.col(ts_col).dt.convert_time_zone("US/Eastern")
    minute_of_day = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)
    angle = (minute_of_day.cast(pl.Float64) / 1440.0) * 2.0 * 3.141592653589793
    return bars.with_columns([
        angle.sin().alias("minute_of_day_sin"),
        angle.cos().alias("minute_of_day_cos"),
    ])
