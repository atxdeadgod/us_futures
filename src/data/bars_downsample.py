"""Downsample 5-sec bars to coarser horizons (1-min, 15-min, etc.).

Two aggregation patterns supported:

1. **Simple column aggregation** (Type 1): per-column rule → first/last/sum/mean/max/min.
   E.g., close → last, volume → sum, cvd_globex → last.

2. **Sub-bar-derived statistics** (Type 2): compute a statistic FROM the time series
   of 5-sec values within the target window. Used for realized moments (vol, skew,
   kurt), which need many sub-samples per aggregation window to be well-estimated.

Usage:
    bars_15m = downsample_bars(bars_5s, target_every="15m")
    realized_15m = realized_moments(bars_5s, target_every="15m")
    combined = bars_15m.join(realized_15m, on="ts")

Both functions use `group_by_dynamic(every=..., closed='left', label='right')` so
bar close timestamps match: a 15-min bar closes at HH:MM where MM ∈ {00,15,30,45}.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Per-column aggregation rules for Type-1 downsampling
# ---------------------------------------------------------------------------
# Default policy per column; not in this dict → defaults to "last".

DEFAULT_AGG_RULES: dict[str, str] = {
    # Identity (first value; stable within a day)
    "root": "first",
    "expiry": "first",
    # OHLC: the classic policy
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    # Volume-like: SUM across sub-bars
    "volume": "sum",
    "dollar_volume": "sum",
    "buys_qty": "sum",
    "sells_qty": "sum",
    "trades_count": "sum",
    "unclassified_count": "sum",
    "implied_volume": "sum",
    "implied_buys": "sum",
    "implied_sells": "sum",
    # L1 snapshot: last of sub-bars
    "bid_close": "last",
    "ask_close": "last",
    "mid_close": "last",
    "spread_abs_close": "last",
    # Spread sub-bar stats: recompute cleanly (mean-of-means is wrong; max/min of
    # maxes is fine). Prefer explicit max/min/mean over sub-bar values. For a
    # downsample these fold correctly because 5-sec spread_mean_sub is itself a
    # within-bar mean; a simple mean-across-sub-bars approximates the parent
    # period mean when sub-bar durations are equal (they are: 5s fixed).
    "spread_mean_sub": "mean",
    "spread_std_sub": "mean",
    "spread_max_sub": "max",
    "spread_min_sub": "min",
    # Book snapshot — last snapshot (closest to target bar close)
    **{f"bid_px_L{k}": "last" for k in range(1, 11)},
    **{f"bid_sz_L{k}": "last" for k in range(1, 11)},
    **{f"bid_ord_L{k}": "last" for k in range(1, 11)},
    **{f"ask_px_L{k}": "last" for k in range(1, 11)},
    **{f"ask_sz_L{k}": "last" for k in range(1, 11)},
    **{f"ask_ord_L{k}": "last" for k in range(1, 11)},
    "book_ts_close": "last",
    # Execution aggregates — SUM
    "eff_spread_sum": "sum",
    "eff_spread_weight": "sum",
    "eff_spread_count": "sum",
    "eff_spread_buy_sum": "sum",
    "eff_spread_buy_weight": "sum",
    "eff_spread_sell_sum": "sum",
    "eff_spread_sell_weight": "sum",
    "n_large_trades": "sum",
    "large_trade_volume": "sum",
    "hidden_absorption_volume": "sum",
    "hidden_absorption_trades": "sum",
    "net_bid_decrement_no_trade_L1": "sum",
    "net_ask_decrement_no_trade_L1": "sum",
    # CVD — LAST (they're cumulative)
    "cvd_globex": "last",
    "cvd_rth": "last",
    "bars_since_rth_reset": "last",
    # Session flags
    "is_rth": "last",
    "is_session_warm": "last",
}


# ---------------------------------------------------------------------------
# Type-1: simple per-column aggregation
# ---------------------------------------------------------------------------

def downsample_bars(
    bars_5s: pl.DataFrame,
    target_every: str = "15m",
    rules: dict[str, str] | None = None,
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Downsample a 5-sec bar frame to `target_every` using per-column rules.

    Columns not listed in `rules` (and not in DEFAULT_AGG_RULES) default to "last".
    """
    if rules is None:
        rules = DEFAULT_AGG_RULES

    aggs = []
    for col, dtype in zip(bars_5s.columns, bars_5s.dtypes):
        if col == ts_col:
            continue
        rule = rules.get(col, "last")
        if rule == "first":
            aggs.append(pl.col(col).first().alias(col))
        elif rule == "last":
            aggs.append(pl.col(col).last().alias(col))
        elif rule == "sum":
            aggs.append(pl.col(col).sum().alias(col))
        elif rule == "mean":
            aggs.append(pl.col(col).mean().alias(col))
        elif rule == "max":
            aggs.append(pl.col(col).max().alias(col))
        elif rule == "min":
            aggs.append(pl.col(col).min().alias(col))
        else:
            raise ValueError(f"Unknown agg rule {rule!r} for col {col!r}")

    result = (
        bars_5s.sort(ts_col)
        .group_by_dynamic(ts_col, every=target_every, closed="left", label="right")
        .agg(aggs)
    )
    return result


# ---------------------------------------------------------------------------
# Type-2: realized moments from sub-bar log returns
# ---------------------------------------------------------------------------

def realized_moments(
    bars_5s: pl.DataFrame,
    target_every: str = "15m",
    ts_col: str = "ts",
    close_col: str = "close",
) -> pl.DataFrame:
    """Compute realized vol / skew / kurt / quarticity / bipower from 5-sec log returns.

    Formulas:
        ret_5s_i   = log(close_i / close_{i-1})
        rv         = Σ ret_5s_i²                   (integrated variance estimator)
        rv_bipower = (π/2) * Σ |r_i| * |r_{i-1}|   (jump-robust per Barndorff-Nielsen-Shephard)
        realized_skew  = skew of ret_5s within the window
        realized_kurt  = excess kurtosis of ret_5s within the window
        realized_quarticity = Σ ret_5s_i⁴ * (n/3)

    Output columns:
        rv_5s, rv_bipower_5s, realized_skew_5s, realized_kurt_5s,
        realized_quarticity_5s, n_subbars
    """
    df = bars_5s.sort(ts_col).with_columns(
        (pl.col(close_col).log() - pl.col(close_col).log().shift(1)).alias("_ret_5s")
    )
    df = df.with_columns(
        [
            pl.col("_ret_5s").abs().alias("_abs_ret"),
            (pl.col("_ret_5s").abs() * pl.col("_ret_5s").abs().shift(1)).alias("_bipower_term"),
        ]
    )
    bipower_const = float(np.pi / 2.0)

    out = (
        df.group_by_dynamic(ts_col, every=target_every, closed="left", label="right")
        .agg(
            [
                (pl.col("_ret_5s") ** 2).sum().alias("rv_5s"),
                (bipower_const * pl.col("_bipower_term").sum()).alias("rv_bipower_5s"),
                pl.col("_ret_5s").skew().alias("realized_skew_5s"),
                pl.col("_ret_5s").kurtosis().alias("realized_kurt_5s"),
                (
                    (pl.col("_ret_5s") ** 4).sum()
                    * (pl.len().cast(pl.Float64) / 3.0)
                ).alias("realized_quarticity_5s"),
                pl.len().alias("n_subbars"),
            ]
        )
    )
    return out


# ---------------------------------------------------------------------------
# Convenience: combined downsample + realized moments
# ---------------------------------------------------------------------------

def downsample_with_moments(
    bars_5s: pl.DataFrame,
    target_every: str = "15m",
    rules: dict[str, str] | None = None,
    ts_col: str = "ts",
    close_col: str = "close",
) -> pl.DataFrame:
    """Downsample + attach realized moments in one call."""
    base = downsample_bars(bars_5s, target_every=target_every, rules=rules, ts_col=ts_col)
    moments = realized_moments(bars_5s, target_every=target_every, ts_col=ts_col, close_col=close_col)
    return base.join(moments, on=ts_col, how="left")
