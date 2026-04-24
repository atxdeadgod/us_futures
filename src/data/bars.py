"""Tick → OHLCV bar aggregation with aggressor volume.

Given trade and quote streams from `ingest.split_trades_quotes`, produce
fixed-frequency bars at arbitrary pandas-style offsets (e.g., "1m", "15m").

Bars that contain zero trades are filled from the last mid-quote (no NaN OHLC)
but retain zero volume and zero aggressor buys/sells.
"""
from __future__ import annotations

import polars as pl


def _mid_quote_series(quotes: pl.DataFrame) -> pl.DataFrame:
    """Return one row per distinct ts with mid = (best_bid + best_ask)/2.

    Quote stream is top-of-book updates, per side. Forward-fill by side then mid.
    """
    bids = (
        quotes.filter(pl.col("side") == "bid")
        .select(["ts", "price"])
        .rename({"price": "bid_price"})
    )
    asks = (
        quotes.filter(pl.col("side") == "ask")
        .select(["ts", "price"])
        .rename({"price": "ask_price"})
    )
    # Merge on ts via outer join + forward-fill both sides, then mid
    merged = (
        bids.join(asks, on="ts", how="full", coalesce=True)
        .sort("ts")
        .with_columns(
            [
                pl.col("bid_price").forward_fill(),
                pl.col("ask_price").forward_fill(),
            ]
        )
        .with_columns(mid=(pl.col("bid_price") + pl.col("ask_price")) / 2)
        .select(["ts", "bid_price", "ask_price", "mid"])
    )
    return merged


def bars_from_trades_quotes(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    every: str = "1m",
) -> pl.DataFrame:
    """Aggregate trades + quotes into OHLCV bars at `every` frequency.

    Columns:
      ts              bar close timestamp (right-closed, right-labeled)
      open/high/low/close   trade-based; for empty bars, mid-quote fallback
      volume          total traded contracts
      buys/sells      signed aggressor-classified volume (AGRESSOR ON BUY / ON SELL)
      trades          trade count
      bid/ask/mid     last quote snapshot at bar close
    """
    # Trade-side bars (may have gaps for quiet bars)
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
                ((pl.col("aggressor_sign") == 1).cast(pl.Int64) * pl.col("quantity"))
                .sum()
                .alias("buys"),
                ((pl.col("aggressor_sign") == -1).cast(pl.Int64) * pl.col("quantity"))
                .sum()
                .alias("sells"),
                pl.len().alias("trades"),
            ]
        )
    )

    # Quote snapshots at same grid — last quote in each bar
    mid = _mid_quote_series(quotes)
    quote_bars = (
        mid.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("bid_price").last().alias("bid"),
                pl.col("ask_price").last().alias("ask"),
                pl.col("mid").last().alias("mid"),
            ]
        )
    )

    # Outer-join trade and quote bars on ts, then fill trade OHLC from mid for empty bars
    joined = (
        trade_bars.join(quote_bars, on="ts", how="full", coalesce=True)
        .sort("ts")
        .with_columns(
            [
                pl.col("bid").forward_fill(),
                pl.col("ask").forward_fill(),
                pl.col("mid").forward_fill(),
            ]
        )
        .with_columns(
            [
                pl.col("open").fill_null(pl.col("mid")),
                pl.col("high").fill_null(pl.col("mid")),
                pl.col("low").fill_null(pl.col("mid")),
                pl.col("close").fill_null(pl.col("mid")),
                pl.col("volume").fill_null(0),
                pl.col("buys").fill_null(0),
                pl.col("sells").fill_null(0),
                pl.col("trades").fill_null(0),
            ]
        )
        .select(
            [
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "buys",
                "sells",
                "trades",
                "bid",
                "ask",
                "mid",
            ]
        )
    )
    return joined
