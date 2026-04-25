"""5-sec bar Phase C: execution-quality aggregates.

Emits per-bar execution-derived features for Tier-1 microstructure (T1.35-T1.37,
T1.23, T1.47, T1.25). Each function takes raw trades/quotes/depth streams and
returns a DataFrame keyed by bar-close `ts`, ready to join onto Phase A bars.

Functions:
  effective_spread_bars     — T1.35, T1.36, T1.37 aggregates
  large_trade_bars          — T1.23 (top-k% of session trade sizes)
  hidden_absorption_bars    — T1.47 (aggressor exceeded L1 size, no price tick)
  quote_direction_bars      — T1.25 prerequisite (per-quote price-move direction counts)

Deferred to Phase D (separate module):
  cancel_proxy_bars       — MBP-10 snapshot-delta cancel attribution (§8.M)
"""
from __future__ import annotations

import polars as pl

from .bars_5sec import BAR_EVERY, l1_stream
from .depth_snap import _asof_strict_backward, split_sides


# ---------------------------------------------------------------------------
# T1.35-T1.37 — Effective spread aggregates
# ---------------------------------------------------------------------------

def effective_spread_bars(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    every: str = BAR_EVERY,
) -> pl.DataFrame:
    """Per-bar effective-spread aggregates.

    Effective spread per trade = 2 * |price - mid_at_trade|, with aggressor-sign
    applied when the aggressor is known:
        aggressor=+1 (buy)  → eff = 2 * (price - mid)   (should be positive)
        aggressor=-1 (sell) → eff = 2 * (mid - price)   (should be positive)
        aggressor= 0        → eff = 2 * |price - mid|   (unsigned fallback)

    Returns: ts, eff_spread_sum, eff_spread_weight, eff_spread_count,
             eff_spread_buy_sum, eff_spread_buy_weight,
             eff_spread_sell_sum, eff_spread_sell_weight.

    Downstream: vwap_eff_spread = eff_spread_sum / eff_spread_weight;
                asymmetry = (buy_sum/buy_weight) - (sell_sum/sell_weight).
    """
    l1 = l1_stream(quotes).select(["ts", "mid"])
    # Asof-strict-backward join trades to mid (§8.E no lookahead)
    trades_mid = _asof_strict_backward(trades.sort("ts"), l1, on="ts")
    # Compute signed eff spread
    trades_mid = trades_mid.with_columns(
        pl.when(pl.col("aggressor_sign") == 0)
        .then(2.0 * (pl.col("price") - pl.col("mid")).abs())
        .otherwise(2.0 * pl.col("aggressor_sign").cast(pl.Float64) * (pl.col("price") - pl.col("mid")))
        .alias("eff_spread")
    )
    # Drop trades before any quote observation (mid is null)
    trades_mid = trades_mid.filter(pl.col("mid").is_not_null())

    bars = (
        trades_mid.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                (pl.col("eff_spread") * pl.col("quantity")).sum().alias("eff_spread_sum"),
                pl.col("quantity").sum().alias("eff_spread_weight"),
                pl.len().alias("eff_spread_count"),
                ((pl.col("aggressor_sign") == 1).cast(pl.Int64) * pl.col("eff_spread") * pl.col("quantity"))
                .sum().alias("eff_spread_buy_sum"),
                ((pl.col("aggressor_sign") == 1).cast(pl.Int64) * pl.col("quantity"))
                .sum().alias("eff_spread_buy_weight"),
                ((pl.col("aggressor_sign") == -1).cast(pl.Int64) * pl.col("eff_spread") * pl.col("quantity"))
                .sum().alias("eff_spread_sell_sum"),
                ((pl.col("aggressor_sign") == -1).cast(pl.Int64) * pl.col("quantity"))
                .sum().alias("eff_spread_sell_weight"),
            ]
        )
    )
    return bars


# ---------------------------------------------------------------------------
# T1.23 — Large trade flags
# ---------------------------------------------------------------------------

def large_trade_bars(
    trades: pl.DataFrame,
    every: str = BAR_EVERY,
    threshold_pct: float = 0.99,
) -> pl.DataFrame:
    """Per-bar large-trade count and volume.

    "Large" = trade size at or above the `threshold_pct` quantile of the day's
    trade-size distribution. Default 0.99 → top 1% of trades.

    Returns: ts, n_large_trades, large_trade_volume.
    """
    if trades.is_empty():
        return pl.DataFrame(
            {"ts": [], "n_large_trades": [], "large_trade_volume": []},
            schema={"ts": trades.schema.get("ts", pl.Datetime("ns", "UTC")),
                    "n_large_trades": pl.Int64, "large_trade_volume": pl.Int64},
        )
    threshold = trades["quantity"].quantile(threshold_pct)
    # Use strict > (not >=) so that when the threshold lands at the modal trade
    # size (common: ES qty=1 is the mode), we don't tag all modal trades as "large".
    # This is the "strictly larger than typical" semantics.
    tagged = trades.with_columns(
        (pl.col("quantity") > threshold).alias("_is_large")
    )
    bars = (
        tagged.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("_is_large").cast(pl.Int64).sum().alias("n_large_trades"),
                (pl.col("_is_large").cast(pl.Int64) * pl.col("quantity")).sum().alias("large_trade_volume"),
            ]
        )
    )
    return bars


# ---------------------------------------------------------------------------
# T1.47 — Hidden / iceberg absorption volume
# ---------------------------------------------------------------------------

def hidden_absorption_bars(
    trades: pl.DataFrame,
    depth: pl.DataFrame,
    every: str = BAR_EVERY,
    only_regular: bool = True,
) -> pl.DataFrame:
    """Per-bar hidden liquidity absorption volume (T1.47).

    For each trade:
      - If aggressor=+1 (buy) → compare to L1 ASK at instant before trade
      - If aggressor=-1 (sell) → compare to L1 BID at instant before trade
      - If aggressor=0 → skip (can't attribute side)
    Check: trade.price matches the L1 price on that side AND trade.quantity
    exceeds the visible L1 size. The overflow (qty − L1_size_pre) is the hidden
    absorption (iceberg-style hidden orders or refill from deeper queues).

    Returns: ts, hidden_absorption_volume, hidden_absorption_trades.
    """
    bid, ask = split_sides(depth, only_regular=only_regular)
    bid_l1 = bid.select(["ts", "bid_px_L1", "bid_sz_L1"]).sort("ts")
    ask_l1 = ask.select(["ts", "ask_px_L1", "ask_sz_L1"]).sort("ts")

    # Strict-backward asof: we want the L1 state STRICTLY BEFORE each trade
    trades_sorted = trades.sort("ts")
    with_ask = _asof_strict_backward(trades_sorted, ask_l1, on="ts")
    with_both = _asof_strict_backward(with_ask, bid_l1, on="ts")

    # Compute hidden absorption volume per trade
    enriched = with_both.with_columns(
        pl.when(
            (pl.col("aggressor_sign") == 1)
            & (pl.col("ask_px_L1").is_not_null())
            & (pl.col("price") == pl.col("ask_px_L1"))
            & (pl.col("quantity") > pl.col("ask_sz_L1"))
        )
        .then(pl.col("quantity") - pl.col("ask_sz_L1"))
        .when(
            (pl.col("aggressor_sign") == -1)
            & (pl.col("bid_px_L1").is_not_null())
            & (pl.col("price") == pl.col("bid_px_L1"))
            & (pl.col("quantity") > pl.col("bid_sz_L1"))
        )
        .then(pl.col("quantity") - pl.col("bid_sz_L1"))
        .otherwise(0)
        .alias("_hidden_vol")
    )

    bars = (
        enriched.group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("_hidden_vol").sum().alias("hidden_absorption_volume"),
                (pl.col("_hidden_vol") > 0).cast(pl.Int64).sum().alias("hidden_absorption_trades"),
            ]
        )
    )
    return bars


# ---------------------------------------------------------------------------
# T1.25 — Quote-movement directionality (per-bar event-direction counts)
# ---------------------------------------------------------------------------

def quote_direction_bars(
    quotes: pl.DataFrame, every: str = BAR_EVERY,
) -> pl.DataFrame:
    """Per-bar counts of quote-update direction events (per side).

    For each side (bid/ask), classify each quote update by comparing its
    price to the previous quote on the same side:
        up   → price strictly higher than prev
        down → price strictly lower than prev
        flat → unchanged or first-of-side

    Returns: ts, bid_up_count, bid_down_count, ask_up_count, ask_down_count.

    Downstream T1.25 quote_movement_directionality computes:
        ((bid_up + ask_down) − (bid_down + ask_up)) / total_directional_events
    Range [-1, 1]: positive = price-improving regime (bid up + ask down).
    """
    bid = (
        quotes.filter(pl.col("side") == "bid")
        .sort("ts")
        .with_columns(pl.col("price").diff().sign().alias("_dir"))
    )
    ask = (
        quotes.filter(pl.col("side") == "ask")
        .sort("ts")
        .with_columns(pl.col("price").diff().sign().alias("_dir"))
    )

    bid_bars = (
        bid.group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg([
            (pl.col("_dir") == 1).cast(pl.Int64).sum().alias("bid_up_count"),
            (pl.col("_dir") == -1).cast(pl.Int64).sum().alias("bid_down_count"),
        ])
    )
    ask_bars = (
        ask.group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg([
            (pl.col("_dir") == 1).cast(pl.Int64).sum().alias("ask_up_count"),
            (pl.col("_dir") == -1).cast(pl.Int64).sum().alias("ask_down_count"),
        ])
    )
    return bid_bars.join(ask_bars, on="ts", how="full", coalesce=True).sort("ts")
