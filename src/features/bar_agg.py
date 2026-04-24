"""Bar-level aggregated flow/book features — operating on our 5-sec bar schema.

feature-factory's `bar/aggregation/*` classes embed the bar-building step into
their compute (`group_by_dynamic` on raw event streams). Since our 5-sec bar
builder has ALREADY aggregated the stream, the features simplify to direct
polars expressions on the bar schema columns.

Covered (simple-from-bar-schema):
  T1.21  order_flow_imbalance
  T1.22  (aggressor_side_volume_ratio) — buys_qty / sells_qty etc. — already cols
  T1.23  large_trade_volume_share
  T1.26  bid_ask_depth_ratio            depth-k
  T1.27  side_weighted_spread (variant using top-of-book)
  Helpers: average_trade_size, rolling_volume_sum, rolling_volume_std

Deferred to Phase E bar-builder extension (need per-quote state in bar):
  T1.24  quote_to_trade            (needs quote_update_count column)
  T1.25  quote_movement_directionality (needs per-quote direction accounting)
  T1.28  side_conditioned_liquidity_shift (needs depth-change-per-event within bar)
  T1.29  liquidity_migration      (same — event-level depth migrations)
"""
from __future__ import annotations

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# T1.21 Order Flow Imbalance (bar-level)
# ---------------------------------------------------------------------------

def order_flow_imbalance(buys_col: pl.Expr, sells_col: pl.Expr) -> pl.Expr:
    """T1.21: (buys - sells) / (buys + sells)."""
    return (buys_col - sells_col) / (buys_col + sells_col + EPS)


# ---------------------------------------------------------------------------
# T1.22 Aggressor Side Volume variants — convenience
# ---------------------------------------------------------------------------

def aggressor_side_ratio(buys_col: pl.Expr, sells_col: pl.Expr) -> pl.Expr:
    """T1.22b: buys / total (returns [0, 1])."""
    return buys_col / (buys_col + sells_col + EPS)


def net_aggressor_volume(buys_col: pl.Expr, sells_col: pl.Expr) -> pl.Expr:
    """T1.22a: buys − sells (signed). Positive = net buy aggression."""
    return buys_col - sells_col


# ---------------------------------------------------------------------------
# T1.23 Large Trade Volume Share
# ---------------------------------------------------------------------------

def large_trade_volume_share(
    large_trade_volume_col: pl.Expr, volume_col: pl.Expr
) -> pl.Expr:
    """T1.23: large_trade_volume / volume. Share of bar volume in top-1% trades."""
    return large_trade_volume_col / (volume_col + EPS)


# ---------------------------------------------------------------------------
# T1.26 Bid-Ask Depth Ratio
# ---------------------------------------------------------------------------

def bid_ask_depth_ratio(levels: int = 5) -> pl.Expr:
    """T1.26: Σ bid_sz_Lk / Σ ask_sz_Lk, k∈1..levels.

    > 1 → bid-heavy book. Uses our bar schema column names.
    """
    bid_cols = [pl.col(f"bid_sz_L{k}") for k in range(1, levels + 1)]
    ask_cols = [pl.col(f"ask_sz_L{k}") for k in range(1, levels + 1)]
    bid_sum = sum(bid_cols[1:], start=bid_cols[0])
    ask_sum = sum(ask_cols[1:], start=ask_cols[0])
    return bid_sum / (ask_sum + EPS)


# ---------------------------------------------------------------------------
# T1.27 Side-Weighted Spread
# ---------------------------------------------------------------------------

def side_weighted_spread_topbook(
    bid_sz_L1: pl.Expr, ask_sz_L1: pl.Expr, spread_abs_col: pl.Expr
) -> pl.Expr:
    """T1.27 simplified: spread weighted by which side is heavier.

    When bid_sz > ask_sz → seller pressure outweighed; spread seen from the
    buy-side is effectively asymmetric. This version uses imbalance as the
    weight: out = spread × (bid - ask) / (bid + ask).
    """
    imbalance = (bid_sz_L1 - ask_sz_L1) / (bid_sz_L1 + ask_sz_L1 + EPS)
    return spread_abs_col * imbalance


# ---------------------------------------------------------------------------
# Volume / trade-count helpers
# ---------------------------------------------------------------------------

def average_trade_size(volume_col: pl.Expr, trades_count_col: pl.Expr) -> pl.Expr:
    """Avg trade size: volume / trades_count."""
    return volume_col / (trades_count_col + EPS)


def rolling_volume_sum(volume_col: pl.Expr, window: int) -> pl.Expr:
    """Rolling-N-bar volume sum (useful for momentum scaling)."""
    return volume_col.rolling_sum(window_size=window)


def rolling_volume_std(volume_col: pl.Expr, window: int) -> pl.Expr:
    """Rolling-N-bar volume std (used by VolumeSurprise variants)."""
    return volume_col.rolling_std(window_size=window)


def rolling_trade_count_mean(trades_count_col: pl.Expr, window: int) -> pl.Expr:
    """Rolling-N-bar average trade count."""
    return trades_count_col.rolling_mean(window_size=window)
