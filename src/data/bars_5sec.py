"""5-sec base bar builder — Phase A (core columns).

Consumes output of ingest.split_trades_quotes() for a single (root, expiry, day)
and emits a dense 5-sec bar DataFrame per the FEATURES.md schema.

Phase A columns (this module):
    Identity           : ts, root, expiry, is_rth, is_session_warm
    OHLCV              : open, high, low, close, volume, dollar_volume
    Aggressor-signed   : buys_qty, sells_qty, trades_count, unclassified_count
    Implied split      : implied_volume, implied_buys, implied_sells
    L1 at close        : bid_close, ask_close, mid_close, spread_abs_close
    Spread sub-bar     : spread_mean_sub, spread_std_sub, spread_max_sub, spread_min_sub
    CVD (dual reset)   : cvd_globex, cvd_rth, bars_since_rth_reset

Phase B (separate module, later): L1-L10 book snapshot, effective spread aggregates,
large-trade flags, cancel proxy, hidden-liquidity tracker.
"""
from __future__ import annotations

import polars as pl

from ..features.engines import cvd_with_dual_reset

BAR_EVERY = "5s"


def l1_stream(quotes: pl.DataFrame) -> pl.DataFrame:
    """Build an L1-mid event stream from the quote stream.

    Quotes arrive as per-side updates (either bid or ask). We forward-fill across
    sides to get a contemporaneous (bid, ask) at each event, then compute mid +
    spread. Output has one row per distinct `ts` with the latest (bid, ask).
    """
    bids = (
        quotes.filter(pl.col("side") == "bid")
        .select(["ts", pl.col("price").alias("bid_price")])
    )
    asks = (
        quotes.filter(pl.col("side") == "ask")
        .select(["ts", pl.col("price").alias("ask_price")])
    )
    merged = (
        bids.join(asks, on="ts", how="full", coalesce=True)
        .sort("ts")
        .with_columns(
            [
                pl.col("bid_price").forward_fill(),
                pl.col("ask_price").forward_fill(),
            ]
        )
        .with_columns(
            mid=(pl.col("bid_price") + pl.col("ask_price")) / 2,
            spread=(pl.col("ask_price") - pl.col("bid_price")),
        )
        .select(["ts", "bid_price", "ask_price", "mid", "spread"])
    )
    return merged


def build_5sec_bars_core(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    root: str,
    expiry: str,
    every: str = BAR_EVERY,
) -> pl.DataFrame:
    """Emit Phase A core columns only. Phase B book-depth enrichment separately."""
    # ---- L1 stream (for mid/spread/quote close aggregates) ----
    l1 = l1_stream(quotes)

    # ---- OHLCV + aggressor aggregation ----
    trade_bars = (
        trades.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("quantity").sum().alias("volume"),
                (pl.col("price") * pl.col("quantity")).sum().alias("dollar_volume"),
                ((pl.col("aggressor_sign") == 1).cast(pl.Int64) * pl.col("quantity")).sum().alias("buys_qty"),
                ((pl.col("aggressor_sign") == -1).cast(pl.Int64) * pl.col("quantity")).sum().alias("sells_qty"),
                pl.len().alias("trades_count"),
                (pl.col("aggressor_sign") == 0).cast(pl.Int64).sum().alias("unclassified_count"),
                # Implied split
                (pl.col("is_implied").cast(pl.Int64) * pl.col("quantity")).sum().alias("implied_volume"),
                ((pl.col("is_implied") & (pl.col("aggressor_sign") == 1)).cast(pl.Int64) * pl.col("quantity")).sum().alias("implied_buys"),
                ((pl.col("is_implied") & (pl.col("aggressor_sign") == -1)).cast(pl.Int64) * pl.col("quantity")).sum().alias("implied_sells"),
            ]
        )
    )

    # ---- Quote stats per bar: last bid/ask/mid + sub-bar spread stats ----
    quote_bars = (
        l1.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("bid_price").last().alias("bid_close"),
                pl.col("ask_price").last().alias("ask_close"),
                pl.col("mid").last().alias("mid_close"),
                pl.col("spread").last().alias("spread_abs_close"),
                pl.col("spread").mean().alias("spread_mean_sub"),
                pl.col("spread").std().alias("spread_std_sub"),
                pl.col("spread").max().alias("spread_max_sub"),
                pl.col("spread").min().alias("spread_min_sub"),
            ]
        )
    )

    # ---- Full outer join, fill empty bars from forward-filled quotes ----
    bars = (
        trade_bars.join(quote_bars, on="ts", how="full", coalesce=True)
        .sort("ts")
        .with_columns(
            [
                pl.col("bid_close").forward_fill(),
                pl.col("ask_close").forward_fill(),
                pl.col("mid_close").forward_fill(),
                pl.col("spread_abs_close").forward_fill(),
            ]
        )
        .with_columns(
            [
                # If a bar had no trades, fill OHLC from last-known mid
                pl.col("open").fill_null(pl.col("mid_close")),
                pl.col("high").fill_null(pl.col("mid_close")),
                pl.col("low").fill_null(pl.col("mid_close")),
                pl.col("close").fill_null(pl.col("mid_close")),
                # Volume/flow cols default to 0 for empty bars
                pl.col("volume").fill_null(0),
                pl.col("dollar_volume").fill_null(0.0),
                pl.col("buys_qty").fill_null(0),
                pl.col("sells_qty").fill_null(0),
                pl.col("trades_count").fill_null(0),
                pl.col("unclassified_count").fill_null(0),
                pl.col("implied_volume").fill_null(0),
                pl.col("implied_buys").fill_null(0),
                pl.col("implied_sells").fill_null(0),
                # Spread sub-stats — null if no quotes in bar (keep as null)
            ]
        )
    )

    # ---- Identity columns ----
    bars = bars.with_columns(
        [
            pl.lit(root).alias("root"),
            pl.lit(expiry).alias("expiry"),
        ]
    )

    # ---- Session flags (in America/New_York for ET-based RTH) ----
    bars = bars.with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").alias("_ts_et")
    )
    bars = bars.with_columns(
        [
            # is_rth: 09:30 <= time < 16:00 ET (cash-equities window)
            (
                (pl.col("_ts_et").dt.hour().cast(pl.Int32) * 60 + pl.col("_ts_et").dt.minute().cast(pl.Int32))
                .is_between(570, 960)  # [09:30, 16:00)
            ).alias("is_rth"),
            # is_session_warm: not within first 30s of Globex reopen at 17:00 ET
            #   Globex daily halt 16:00-17:00 ET. First 30s post-reopen = 17:00:00 - 17:00:30 ET.
            # We mark NOT warm if within that window; otherwise warm.
            (
                ~(
                    (pl.col("_ts_et").dt.hour().cast(pl.Int32) == 17)
                    & (pl.col("_ts_et").dt.minute().cast(pl.Int32) == 0)
                    & (pl.col("_ts_et").dt.second().cast(pl.Int32) <= 30)
                )
            ).alias("is_session_warm"),
        ]
    ).drop("_ts_et")

    # ---- CVD dual-reset ----
    bars = cvd_with_dual_reset(bars)

    # ---- Column order per FEATURES.md schema ----
    col_order = [
        "ts", "root", "expiry", "is_rth", "is_session_warm",
        "open", "high", "low", "close", "volume", "dollar_volume",
        "buys_qty", "sells_qty", "trades_count", "unclassified_count",
        "implied_volume", "implied_buys", "implied_sells",
        "bid_close", "ask_close", "mid_close", "spread_abs_close",
        "spread_mean_sub", "spread_std_sub", "spread_max_sub", "spread_min_sub",
        "cvd_globex", "cvd_rth", "bars_since_rth_reset",
    ]
    return bars.select([c for c in col_order if c in bars.columns])
