"""Top-of-book (L1) features — pure polars expressions.

Extracted from feature-factory (commit frozen 2026-01) into standalone
expressions. No framework, no BaseFeature — just functions that return
polars expressions you combine via `.with_columns()`.

Covers FEATURES.md entries:
    T1.01 Microprice
    T1.02 MicropriceDrift
    T1.03 MidPriceReturn
    T1.04 OrderImbalance
    T1.05 SpreadAbs
    T1.06 SpreadRelBps
    T1.07 SpreadVolatilityRatio
    T1.08 QuoteSlopeProxy
    T2.12 VolatilityOfVolatility
    T2.13 UpVolatility
    T2.14 DownVolatility
    T2.15 VolDirectionRatio
    T2.16 TickVolatility
    T2.17 TickReturnHigherMoments (skew, kurt)
    T2.19 JumpIntensity

Convention: each function returns `pl.Expr`. Caller applies via
    df.with_columns(l1.microprice(...).alias("microprice"))
"""
from __future__ import annotations

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# Price-level derivatives
# ---------------------------------------------------------------------------

def mid_price(bid_px: pl.Expr, ask_px: pl.Expr) -> pl.Expr:
    """T1.03 input: (bid + ask) / 2."""
    return (bid_px + ask_px) / 2.0


def microprice(bid_px: pl.Expr, ask_px: pl.Expr, bid_sz: pl.Expr, ask_sz: pl.Expr) -> pl.Expr:
    """T1.01 Microprice: size-weighted inner value.
        (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)
    """
    return (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz + EPS)


def microprice_drift(microprice_col: pl.Expr, shift: int = 1) -> pl.Expr:
    """T1.02 MicropriceDrift: log(microprice / microprice.shift(shift)) with forward-fill."""
    ff = microprice_col.forward_fill()
    return (ff / ff.shift(shift)).log()


def mid_price_return(mid_col: pl.Expr) -> pl.Expr:
    """T1.03 MidPriceReturn: log(mid / mid.shift(1))."""
    return (mid_col / mid_col.shift(1)).log()


# ---------------------------------------------------------------------------
# Imbalance
# ---------------------------------------------------------------------------

def order_imbalance(bid_sz: pl.Expr, ask_sz: pl.Expr, window: int = 1) -> pl.Expr:
    """T1.04 OrderImbalance: rolling mean of bid_share.
    bid_share = bid_sz / (bid_sz + ask_sz)
    """
    raw = bid_sz / (bid_sz + ask_sz + EPS)
    if window > 1:
        return raw.rolling_mean(window_size=window)
    return raw


# ---------------------------------------------------------------------------
# Spread family
# ---------------------------------------------------------------------------

def spread_abs(bid_px: pl.Expr, ask_px: pl.Expr) -> pl.Expr:
    """T1.05: ask - bid."""
    return ask_px - bid_px


def spread_rel_bps(spread_abs_col: pl.Expr, mid_col: pl.Expr) -> pl.Expr:
    """T1.06: spread / mid × 10000."""
    return (spread_abs_col / (mid_col + EPS)) * 10_000


def spread_volatility_ratio(spread_rel_bps_col: pl.Expr, tick_vol_col: pl.Expr) -> pl.Expr:
    """T1.07: relative spread normalized by tick vol."""
    return spread_rel_bps_col / (tick_vol_col + EPS)


def quote_slope_proxy(spread_abs_col: pl.Expr, bid_sz: pl.Expr, ask_sz: pl.Expr) -> pl.Expr:
    """T1.08: book slope proxy = spread / (bid_sz + ask_sz).
    Higher slope = less depth per unit price → more price impact.
    """
    return spread_abs_col / (bid_sz + ask_sz + EPS)


# ---------------------------------------------------------------------------
# Volatility family (operates on mid-price-return)
# ---------------------------------------------------------------------------

def tick_volatility(ret_col: pl.Expr, window: int) -> pl.Expr:
    """T2.16: rolling std of mid-return over `window`."""
    return ret_col.rolling_std(window_size=window)


def up_volatility(ret_col: pl.Expr, window: int) -> pl.Expr:
    """T2.13: rolling std of positive returns only."""
    positive = pl.when(ret_col > 0).then(ret_col).otherwise(None)
    return positive.rolling_std(window_size=window).fill_null(0.0)


def down_volatility(ret_col: pl.Expr, window: int) -> pl.Expr:
    """T2.14: rolling std of negative returns only."""
    negative = pl.when(ret_col < 0).then(ret_col).otherwise(None)
    return negative.rolling_std(window_size=window).fill_null(0.0)


def vol_direction_ratio(down_vol_col: pl.Expr, up_vol_col: pl.Expr) -> pl.Expr:
    """T2.15: down-vol / up-vol (> 1 → bearish skew in vol)."""
    return down_vol_col / (up_vol_col + EPS)


def volatility_of_volatility(tick_vol_col: pl.Expr, window: int) -> pl.Expr:
    """T2.12: rolling std of tick vol (vol-of-vol)."""
    return tick_vol_col.rolling_std(window_size=window)


# ---------------------------------------------------------------------------
# Higher moments (rolling skew + rolling kurtosis from returns)
# ---------------------------------------------------------------------------

def tick_return_skew(ret_col: pl.Expr, window: int) -> pl.Expr:
    """T2.17a: rolling skew of returns."""
    return ret_col.rolling_skew(window_size=window, bias=False)


def tick_return_kurtosis(ret_col: pl.Expr, window: int) -> pl.Expr:
    """T2.17b: rolling kurtosis (Fisher = excess) of returns."""
    return ret_col.rolling_kurtosis(window_size=window, fisher=True)


# ---------------------------------------------------------------------------
# Jump intensity
# ---------------------------------------------------------------------------

def jump_intensity(
    ret_col: pl.Expr,
    tick_vol_col: pl.Expr,
    intensity_window: int,
    jump_threshold: float = 3.0,
) -> pl.Expr:
    """T2.19: rolling count of bars where |return| exceeds jump_threshold × prev-vol.
    Per Barndorff-Nielsen-Shephard style jump flag, aggregated over N bars.
    """
    is_jump = (ret_col.abs() > jump_threshold * tick_vol_col.shift(1)).cast(pl.Int32)
    return is_jump.rolling_sum(window_size=intensity_window, min_samples=1)
