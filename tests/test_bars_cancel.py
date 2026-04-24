"""Unit tests for src/data/bars_cancel.py Phase D cancel proxy (L1 only)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_cancel import cancel_proxy_bars


def _mk_trades(rows):
    if not rows:
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime("ns", "UTC"),
                "price": pl.Float64,
                "quantity": pl.Int64,
                "aggressor_sign": pl.Int8,
                "is_implied": pl.Boolean,
            }
        )
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("ts").cast(pl.Datetime("ns", "UTC")),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Int64),
            pl.col("aggressor_sign").cast(pl.Int8),
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


def _bid(ts, s1=10, **kw):
    r = {"ts": ts, "Side": "B", "Flags": 0}
    for k in range(1, 11):
        r[f"L{k}Price"] = kw.get(f"p{k}", 5000.0 if k == 1 else 0.0)
        r[f"L{k}Size"] = kw.get(f"s{k}", s1 if k == 1 else 0)
        r[f"L{k}Orders"] = 0
    return r


def _ask(ts, s1=10, **kw):
    r = {"ts": ts, "Side": "S", "Flags": 0}
    for k in range(1, 11):
        r[f"L{k}Price"] = kw.get(f"p{k}", 5000.25 if k == 1 else 0.0)
        r[f"L{k}Size"] = kw.get(f"s{k}", s1 if k == 1 else 0)
        r[f"L{k}Orders"] = 0
    return r


def test_bid_size_drop_all_explained_by_trades():
    """Bid L1 goes 50 → 20 (drop=30); 30 aggressor-sell trades → cancel = 0."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([_bid(t0, s1=50), _bid(t0 + timedelta(seconds=3), s1=20), _ask(t0, s1=40)])
    trades = _mk_trades(
        [{"ts": t0 + timedelta(seconds=2), "price": 5000.0, "quantity": 30, "aggressor_sign": -1, "is_implied": False}]
    )
    bars = cancel_proxy_bars(trades, depth)
    # Find the bar covering [t0, t0+5s)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_bid_decrement_no_trade_L1"] == 0
    assert row["net_ask_decrement_no_trade_L1"] == 0


def test_bid_size_drop_all_cancel_no_trades():
    """Bid L1 drops 50 → 20 with NO trades → full 30 attributed to cancel."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([_bid(t0, s1=50), _bid(t0 + timedelta(seconds=3), s1=20), _ask(t0, s1=40)])
    trades = _mk_trades([])  # no trades at all
    bars = cancel_proxy_bars(trades, depth)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_bid_decrement_no_trade_L1"] == 30


def test_bid_partial_explanation():
    """Bid L1 drops 100 → 20 (drop=80); 50 sell-agg trades → cancel = 30."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([_bid(t0, s1=100), _bid(t0 + timedelta(seconds=3), s1=20), _ask(t0, s1=40)])
    trades = _mk_trades(
        [{"ts": t0 + timedelta(seconds=2), "price": 5000.0, "quantity": 50, "aggressor_sign": -1, "is_implied": False}]
    )
    bars = cancel_proxy_bars(trades, depth)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_bid_decrement_no_trade_L1"] == 30  # 80 - 50


def test_ask_size_drop_cancel():
    """Symmetric test on ask side: 100 → 30 with 40 lift-ask trades → cancel = 30."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([_ask(t0, s1=100), _ask(t0 + timedelta(seconds=3), s1=30), _bid(t0, s1=40)])
    trades = _mk_trades(
        [{"ts": t0 + timedelta(seconds=2), "price": 5000.25, "quantity": 40, "aggressor_sign": 1, "is_implied": False}]
    )
    bars = cancel_proxy_bars(trades, depth)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_ask_decrement_no_trade_L1"] == 30
    assert row["net_bid_decrement_no_trade_L1"] == 0


def test_size_increase_yields_zero_cancel():
    """If L1 size INCREASES during the bar, cancel proxy is 0 (not negative)."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([_bid(t0, s1=20), _bid(t0 + timedelta(seconds=3), s1=80), _ask(t0, s1=40)])
    trades = _mk_trades([])
    bars = cancel_proxy_bars(trades, depth)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_bid_decrement_no_trade_L1"] == 0


def test_trades_exceed_decrement_still_zero():
    """If trade volume exceeds decrement (bid grew back), cancel proxy is 0."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth([_bid(t0, s1=50), _bid(t0 + timedelta(seconds=3), s1=40), _ask(t0, s1=40)])
    # 30 sell-trade volume but bid only dropped 10
    trades = _mk_trades(
        [{"ts": t0 + timedelta(seconds=2), "price": 5000.0, "quantity": 30, "aggressor_sign": -1, "is_implied": False}]
    )
    bars = cancel_proxy_bars(trades, depth)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_bid_decrement_no_trade_L1"] == 0


def test_empty_trades_nonzero_cancel_emitted():
    """Empty trades frame + depth events → cancels computed from depth alone."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth(
        [
            _bid(t0, s1=100),
            _bid(t0 + timedelta(seconds=3), s1=40),
            _ask(t0, s1=80),
            _ask(t0 + timedelta(seconds=4), s1=30),
        ]
    )
    trades = _mk_trades([])
    bars = cancel_proxy_bars(trades, depth)
    row = bars.filter(pl.col("ts") == t0 + timedelta(seconds=5)).row(0, named=True)
    assert row["net_bid_decrement_no_trade_L1"] == 60
    assert row["net_ask_decrement_no_trade_L1"] == 50
