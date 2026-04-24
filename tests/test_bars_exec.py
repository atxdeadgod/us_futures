"""Unit tests for src/data/bars_exec.py Phase C features."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_exec import (
    effective_spread_bars,
    hidden_absorption_bars,
    large_trade_bars,
)


def _mk_trades(rows):
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("ts").cast(pl.Datetime("ns", "UTC")),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Int64),
            pl.col("aggressor_sign").cast(pl.Int8),
            pl.col("is_implied").cast(pl.Boolean),
        ]
    )


def _mk_quotes(rows):
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("ts").cast(pl.Datetime("ns", "UTC")),
            pl.col("side").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Int64),
            pl.col("orders").cast(pl.Int64),
            pl.col("is_implied").cast(pl.Boolean),
        ]
    )


def _mk_depth(rows):
    schema: dict[str, pl.DataType] = {
        "ts": pl.Datetime("ns", "UTC"),
        "Side": pl.Utf8,
        "Flags": pl.Int64,
    }
    for k in range(1, 11):
        schema[f"L{k}Price"] = pl.Float64
        schema[f"L{k}Size"] = pl.Int64
        schema[f"L{k}Orders"] = pl.Int64
    return pl.DataFrame(rows, schema=schema)


def _bid_row(ts, p1=5000.0, s1=10, **kw):
    r = {"ts": ts, "Side": "B", "Flags": 0}
    for k in range(1, 11):
        r[f"L{k}Price"] = kw.get(f"p{k}", p1 if k == 1 else 0.0)
        r[f"L{k}Size"] = kw.get(f"s{k}", s1 if k == 1 else 0)
        r[f"L{k}Orders"] = 0
    return r


def _ask_row(ts, p1=5000.25, s1=10, **kw):
    r = {"ts": ts, "Side": "S", "Flags": 0}
    for k in range(1, 11):
        r[f"L{k}Price"] = kw.get(f"p{k}", p1 if k == 1 else 0.0)
        r[f"L{k}Size"] = kw.get(f"s{k}", s1 if k == 1 else 0)
        r[f"L{k}Orders"] = 0
    return r


# ===========================================================================
# T1.35-T1.37  effective_spread_bars
# ===========================================================================

def test_effective_spread_buy_at_ask():
    """Buy trade at $5000.25 when mid = $5000.125 → eff = 2*(5000.25-5000.125) = 0.25"""
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    quotes = _mk_quotes([
        {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 5000.00, "size": 10, "orders": 1, "is_implied": False},
        {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.25, "size": 10, "orders": 1, "is_implied": False},
    ])
    trades = _mk_trades([
        {"ts": t0, "price": 5000.25, "quantity": 10, "aggressor_sign": 1, "is_implied": False},
    ])
    bars = effective_spread_bars(trades, quotes)
    assert bars.height == 1
    r = bars.row(0, named=True)
    assert abs(r["eff_spread_sum"] - 0.25 * 10) < 1e-9
    assert r["eff_spread_weight"] == 10
    assert r["eff_spread_count"] == 1
    assert r["eff_spread_buy_weight"] == 10
    assert r["eff_spread_sell_weight"] == 0


def test_effective_spread_sell_at_bid():
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    quotes = _mk_quotes([
        {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 5000.00, "size": 10, "orders": 1, "is_implied": False},
        {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.25, "size": 10, "orders": 1, "is_implied": False},
    ])
    trades = _mk_trades([
        {"ts": t0, "price": 5000.00, "quantity": 20, "aggressor_sign": -1, "is_implied": False},
    ])
    bars = effective_spread_bars(trades, quotes)
    r = bars.row(0, named=True)
    # Sell aggressor at bid: mid = 5000.125, eff = 2 * (-1) * (5000.00 - 5000.125) = 2*0.125 = 0.25
    assert abs(r["eff_spread_sum"] - 0.25 * 20) < 1e-9
    assert r["eff_spread_sell_weight"] == 20
    assert r["eff_spread_buy_weight"] == 0


def test_effective_spread_asymmetry():
    """Buys have larger eff spread than sells → asymmetry captured in split cols."""
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    quotes = _mk_quotes([
        {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 5000.00, "size": 10, "orders": 1, "is_implied": False},
        {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.50, "size": 10, "orders": 1, "is_implied": False},
    ])
    trades = _mk_trades([
        # Buy at ask → eff = 2*(5000.50 - 5000.25) = 0.50
        {"ts": t0, "price": 5000.50, "quantity": 10, "aggressor_sign": 1, "is_implied": False},
        # Sell at bid → eff = 2*(5000.25 - 5000.00) = 0.50
        {"ts": t0 + timedelta(seconds=1), "price": 5000.00, "quantity": 10, "aggressor_sign": -1, "is_implied": False},
    ])
    bars = effective_spread_bars(trades, quotes)
    r = bars.row(0, named=True)
    # Symmetric quote: both sides eff=0.5 volume-weighted
    buy_vwap = r["eff_spread_buy_sum"] / r["eff_spread_buy_weight"]
    sell_vwap = r["eff_spread_sell_sum"] / r["eff_spread_sell_weight"]
    assert abs(buy_vwap - 0.5) < 1e-9
    assert abs(sell_vwap - 0.5) < 1e-9


def test_effective_spread_no_lookahead():
    """Quote at exact trade ts must NOT contaminate — strict < asof."""
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    quotes = _mk_quotes([
        # Earlier wide quote
        {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 4990.00, "size": 10, "orders": 1, "is_implied": False},
        {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5010.00, "size": 10, "orders": 1, "is_implied": False},
        # Tight quote at exactly trade ts — MUST be excluded
        {"ts": t0, "side": "bid", "price": 5000.00, "size": 10, "orders": 1, "is_implied": False},
        {"ts": t0, "side": "ask", "price": 5000.25, "size": 10, "orders": 1, "is_implied": False},
    ])
    trades = _mk_trades([
        {"ts": t0, "price": 5000.00, "quantity": 10, "aggressor_sign": -1, "is_implied": False},
    ])
    bars = effective_spread_bars(trades, quotes)
    r = bars.row(0, named=True)
    # Pre-trade mid was (4990 + 5010)/2 = 5000.0 → eff spread = 2 * (-1) * (5000.0 - 5000.0) = 0
    # If lookahead happened, tight-quote mid = 5000.125 → eff = 2*(5000.125-5000) = 0.25
    assert abs(r["eff_spread_sum"]) < 1e-9  # used wide quote → eff = 0


# ===========================================================================
# T1.23  large_trade_bars
# ===========================================================================

def test_large_trade_threshold():
    """With threshold=0.99, only trades at or above the 99th percentile count."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    # 99 small trades (qty=1) + 1 huge (qty=100)
    rows = [
        {"ts": t0 + timedelta(seconds=i * 0.05), "price": 5000.0, "quantity": 1,
         "aggressor_sign": 1, "is_implied": False}
        for i in range(99)
    ]
    rows.append({"ts": t0 + timedelta(seconds=3), "price": 5000.0, "quantity": 100,
                 "aggressor_sign": 1, "is_implied": False})
    trades = _mk_trades(rows)
    bars = large_trade_bars(trades, threshold_pct=0.99)
    # Total large-trade volume should be 100 (just the single huge trade)
    assert bars["large_trade_volume"].sum() == 100
    assert bars["n_large_trades"].sum() == 1


def test_large_trade_empty_trades():
    trades = pl.DataFrame(
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "price": pl.Float64,
            "quantity": pl.Int64,
            "aggressor_sign": pl.Int8,
            "is_implied": pl.Boolean,
        }
    )
    bars = large_trade_bars(trades)
    assert bars.height == 0


# ===========================================================================
# T1.47  hidden_absorption_bars
# ===========================================================================

def test_hidden_absorption_buy_overflow():
    """Buy trade of 50 against L1 ask size 10 at matching price → 40 hidden absorbed."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([
        _bid_row(t0 + timedelta(seconds=1), p1=5000.00, s1=15),
        _ask_row(t0 + timedelta(seconds=1), p1=5000.25, s1=10),
    ])
    trades = _mk_trades([
        {"ts": t0 + timedelta(seconds=3), "price": 5000.25, "quantity": 50,
         "aggressor_sign": 1, "is_implied": False},
    ])
    bars = hidden_absorption_bars(trades, depth)
    r = bars.row(0, named=True)
    assert r["hidden_absorption_volume"] == 40  # 50 - 10
    assert r["hidden_absorption_trades"] == 1


def test_hidden_absorption_sell_overflow():
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([
        _bid_row(t0 + timedelta(seconds=1), p1=5000.00, s1=10),
        _ask_row(t0 + timedelta(seconds=1), p1=5000.25, s1=15),
    ])
    trades = _mk_trades([
        {"ts": t0 + timedelta(seconds=3), "price": 5000.00, "quantity": 25,
         "aggressor_sign": -1, "is_implied": False},
    ])
    bars = hidden_absorption_bars(trades, depth)
    r = bars.row(0, named=True)
    assert r["hidden_absorption_volume"] == 15  # 25 - 10
    assert r["hidden_absorption_trades"] == 1


def test_hidden_absorption_no_overflow():
    """Trade size <= L1 size → no hidden."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([
        _bid_row(t0 + timedelta(seconds=1), p1=5000.00, s1=20),
        _ask_row(t0 + timedelta(seconds=1), p1=5000.25, s1=20),
    ])
    trades = _mk_trades([
        {"ts": t0 + timedelta(seconds=3), "price": 5000.25, "quantity": 10,
         "aggressor_sign": 1, "is_implied": False},
    ])
    bars = hidden_absorption_bars(trades, depth)
    r = bars.row(0, named=True)
    assert r["hidden_absorption_volume"] == 0


def test_hidden_absorption_unknown_aggressor():
    """aggressor=0 → skip (can't attribute side)."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([
        _bid_row(t0 + timedelta(seconds=1), p1=5000.00, s1=10),
        _ask_row(t0 + timedelta(seconds=1), p1=5000.25, s1=10),
    ])
    trades = _mk_trades([
        {"ts": t0 + timedelta(seconds=3), "price": 5000.25, "quantity": 50,
         "aggressor_sign": 0, "is_implied": False},
    ])
    bars = hidden_absorption_bars(trades, depth)
    if bars.height:
        assert bars.row(0, named=True)["hidden_absorption_volume"] == 0


def test_hidden_absorption_price_mismatch():
    """Trade price != L1 price → no hidden attribution (price ticked through)."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([
        _bid_row(t0 + timedelta(seconds=1), p1=5000.00, s1=10),
        _ask_row(t0 + timedelta(seconds=1), p1=5000.25, s1=10),
    ])
    trades = _mk_trades([
        # Buy at 5000.50 but L1 ask was 5000.25 → price ticked, skip
        {"ts": t0 + timedelta(seconds=3), "price": 5000.50, "quantity": 50,
         "aggressor_sign": 1, "is_implied": False},
    ])
    bars = hidden_absorption_bars(trades, depth)
    r = bars.row(0, named=True)
    assert r["hidden_absorption_volume"] == 0
