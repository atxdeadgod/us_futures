"""Causal EMA-smoothed variants of base microstructure features.

A handful of base features are noisy bar-to-bar but the underlying signal
persists over multiple bars (volume_surprise, OFI, realized_vol changes,
spread regime shifts). Smoothing with a causal EMA reveals the slower-moving
signal without introducing future leakage.

Convention:
    out_col = f"{value_col}_ema_s{span}"

Spans of 10 and 30 (at 15-min bars) correspond to ~2.5 hours and ~7.5 hours
of decay — short enough to react within a session, long enough to filter
single-bar noise.

All EMAs are computed via polars `Expr.ewm_mean(span=..., adjust=False,
ignore_nulls=False)` which is the standard causal recursion:
    s_t = (1 − α)·s_{t−1} + α·x_t,  α = 2/(span+1)

`adjust=False` so that we don't apply the bias-correction factor
(1−(1−α)^{n+1}) — that factor is fine for one-shot EWMA but adds lookahead
in a streaming context if upstream callers re-window.
"""
from __future__ import annotations

import polars as pl

DEFAULT_SPANS: tuple[int, ...] = (10, 30)

DEFAULT_VALUE_COLS: tuple[str, ...] = (
    "abs_log_return",
    "ofi",
    "cvd_change",
    "spread_to_mid_bps",
    "vol_surprise_w20",
    "vol_surprise_w60",
    "realized_vol_w20",
    "log_volume",
)


def attach_ema_smoothed(
    bars: pl.DataFrame,
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
    spans: tuple[int, ...] = DEFAULT_SPANS,
) -> pl.DataFrame:
    """For each (value_col × span), append a causal-EMA-smoothed column.

    Skips a value_col silently if it doesn't exist on the frame; this keeps
    the function safe to call on partial-feature frames.
    """
    available = [c for c in value_cols if c in bars.columns]
    if not available:
        return bars
    aggs = []
    for c in available:
        for span in spans:
            aggs.append(
                pl.col(c).ewm_mean(span=span, adjust=False, ignore_nulls=False)
                    .alias(f"{c}_ema_s{span}")
            )
    return bars.with_columns(aggs)
