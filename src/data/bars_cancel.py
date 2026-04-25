"""5-sec bar Phase D: cancel-proxy from MBP-10 snapshot deltas (T1.43 support).

This is the INFERRED cancel feature per §8.M. We don't have MBO (order-by-order)
data — only periodic book snapshots — so we approximate cancels by attributing
unexplained-size-drops to cancellations after accounting for matching trades.

V1 approximation (this module):
    L1 ONLY. For each bar and each side:
      bid_size_decrement   = max(0, first_bid_sz_L1_in_bar - last_bid_sz_L1_in_bar)
      bid_trades_vol       = sum(aggressor_sign == -1 ? quantity : 0)   # trades hit bid
      net_bid_decrement_no_trade_L1 = max(0, bid_size_decrement - bid_trades_vol)

    Symmetric on ask (aggressor=+1 hits ask).

    Assumptions (limitations):
    - We only look at first vs last snapshot of the bar; size oscillations within
      the bar are ignored (conservative — may underestimate true cancel volume).
    - Trade price-level matching is approximate: we count ALL sell-aggressor
      volume as potential "hit the bid", even if trades were at a different
      price. This overestimates trade attribution, which in turn UNDER-estimates
      the cancel residual. That's the safer direction for a noisy signal.
    - L2-L5 decrements deferred to V2 where we'd need proper price-level
      attribution across L1 promotions/demotions.

Output columns (attached to 5-sec bar schema):
    net_bid_decrement_no_trade_L1
    net_ask_decrement_no_trade_L1
    bid_sz_L1_delta_signed         (last - first; T1.28 prereq)
    ask_sz_L1_delta_signed         (last - first; T1.28 prereq)
    hit_bid_vol                    (sell-aggressor volume; T1.28 normalizer)
    lift_ask_vol                   (buy-aggressor volume; T1.28 normalizer)
"""
from __future__ import annotations

import polars as pl

from .bars_5sec import BAR_EVERY
from .depth_snap import split_sides


def cancel_proxy_bars(
    trades: pl.DataFrame,
    depth: pl.DataFrame,
    every: str = BAR_EVERY,
    only_regular: bool = True,
) -> pl.DataFrame:
    """Per-bar L1 cancel-proxy columns (bid & ask).

    Returns DataFrame indexed by bar-close `ts` with:
        net_bid_decrement_no_trade_L1, net_ask_decrement_no_trade_L1

    Bars with no depth events (rare) emit 0.
    """
    bid, ask = split_sides(depth, only_regular=only_regular)

    # --- Bid side: first & last L1 size per bar ---
    bid_bar_stats = (
        bid.select(["ts", "bid_sz_L1"])
        .sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("bid_sz_L1").first().alias("bid_sz_L1_first"),
                pl.col("bid_sz_L1").last().alias("bid_sz_L1_last"),
            ]
        )
    )

    # --- Ask side: first & last L1 size per bar ---
    ask_bar_stats = (
        ask.select(["ts", "ask_sz_L1"])
        .sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("ask_sz_L1").first().alias("ask_sz_L1_first"),
                pl.col("ask_sz_L1").last().alias("ask_sz_L1_last"),
            ]
        )
    )

    # --- Trades per bar: volume hitting bid vs lifting ask ---
    trade_bar_vols = (
        trades.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                ((pl.col("aggressor_sign") == -1).cast(pl.Int64) * pl.col("quantity"))
                .sum().alias("hit_bid_vol"),
                ((pl.col("aggressor_sign") == 1).cast(pl.Int64) * pl.col("quantity"))
                .sum().alias("lift_ask_vol"),
            ]
        )
    )

    # --- Merge all by ts and compute net decrement not explained by trades ---
    out = (
        bid_bar_stats.join(ask_bar_stats, on="ts", how="full", coalesce=True)
        .join(trade_bar_vols, on="ts", how="full", coalesce=True)
        .sort("ts")
        .with_columns(
            [
                pl.col("hit_bid_vol").fill_null(0),
                pl.col("lift_ask_vol").fill_null(0),
            ]
        )
        .with_columns(
            [
                # bid: size decrement not matched by hit-bid trade volume
                pl.max_horizontal(
                    pl.lit(0, dtype=pl.Int64),
                    pl.col("bid_sz_L1_first") - pl.col("bid_sz_L1_last") - pl.col("hit_bid_vol"),
                ).alias("net_bid_decrement_no_trade_L1"),
                # ask: size decrement not matched by lift-ask trade volume
                pl.max_horizontal(
                    pl.lit(0, dtype=pl.Int64),
                    pl.col("ask_sz_L1_first") - pl.col("ask_sz_L1_last") - pl.col("lift_ask_vol"),
                ).alias("net_ask_decrement_no_trade_L1"),
                # T1.28: signed depth deltas (last - first) per bar; combined with
                # aggressor-side volume downstream gives "side-conditioned liquidity
                # response". Positive = depth grew within bar; negative = depth shrank.
                (pl.col("bid_sz_L1_last") - pl.col("bid_sz_L1_first")).alias("bid_sz_L1_delta_signed"),
                (pl.col("ask_sz_L1_last") - pl.col("ask_sz_L1_first")).alias("ask_sz_L1_delta_signed"),
            ]
        )
        .select([
            "ts",
            "net_bid_decrement_no_trade_L1", "net_ask_decrement_no_trade_L1",
            "bid_sz_L1_delta_signed", "ask_sz_L1_delta_signed",
            "hit_bid_vol", "lift_ask_vol",
        ])
    )
    return out
