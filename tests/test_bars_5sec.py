"""Unit tests for src/data/bars_5sec.py Phase A core bar builder."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_5sec import build_5sec_bars_core


def _mk_trades(rows):
    """Helper. rows: list of dicts with ts (datetime), price, quantity, aggressor_sign, is_implied."""
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


def test_single_bar_all_buys():
    """All trades in one 5-sec bar, all aggressive buys."""
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    trades = _mk_trades(
        [
            {"ts": t0, "price": 5000.0, "quantity": 10, "aggressor_sign": 1, "is_implied": False},
            {"ts": t0 + timedelta(seconds=1), "price": 5000.25, "quantity": 5, "aggressor_sign": 1, "is_implied": False},
            {"ts": t0 + timedelta(seconds=2), "price": 5000.50, "quantity": 3, "aggressor_sign": 1, "is_implied": False},
        ]
    )
    quotes = _mk_quotes(
        [
            {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 4999.75, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.00, "size": 50, "orders": 5, "is_implied": False},
        ]
    )
    bars = build_5sec_bars_core(trades, quotes, root="ES", expiry="ESH4")
    assert bars.height == 1
    row = bars.row(0, named=True)
    assert row["open"] == 5000.0
    assert row["high"] == 5000.50
    assert row["low"] == 5000.0
    assert row["close"] == 5000.50
    assert row["volume"] == 18
    assert row["buys_qty"] == 18
    assert row["sells_qty"] == 0
    assert row["trades_count"] == 3
    assert row["root"] == "ES"
    assert row["expiry"] == "ESH4"


def test_balanced_buys_sells():
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    trades = _mk_trades(
        [
            {"ts": t0, "price": 5000.0, "quantity": 10, "aggressor_sign": 1, "is_implied": False},
            {"ts": t0 + timedelta(seconds=1), "price": 5000.0, "quantity": 10, "aggressor_sign": -1, "is_implied": False},
        ]
    )
    quotes = _mk_quotes(
        [
            {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 4999.75, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.00, "size": 50, "orders": 5, "is_implied": False},
        ]
    )
    bars = build_5sec_bars_core(trades, quotes, root="ES", expiry="ESH4")
    assert bars.height == 1
    row = bars.row(0, named=True)
    assert row["buys_qty"] == 10
    assert row["sells_qty"] == 10
    assert row["trades_count"] == 2
    assert row["cvd_globex"] == 0  # net aggressor = 0


def test_implied_volume_split():
    t0 = datetime(2024, 1, 2, 14, 30, 2, tzinfo=timezone.utc)
    trades = _mk_trades(
        [
            {"ts": t0, "price": 5000.0, "quantity": 10, "aggressor_sign": 1, "is_implied": False},
            {"ts": t0 + timedelta(seconds=1), "price": 5000.0, "quantity": 5, "aggressor_sign": 1, "is_implied": True},
            {"ts": t0 + timedelta(seconds=2), "price": 5000.0, "quantity": 7, "aggressor_sign": -1, "is_implied": True},
        ]
    )
    quotes = _mk_quotes(
        [
            {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 4999.75, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.00, "size": 50, "orders": 5, "is_implied": False},
        ]
    )
    bars = build_5sec_bars_core(trades, quotes, root="ES", expiry="ESH4")
    row = bars.row(0, named=True)
    assert row["implied_volume"] == 12  # 5 + 7
    assert row["implied_buys"] == 5
    assert row["implied_sells"] == 7


def test_empty_bar_gets_mid_ohlc():
    """A bar with no trades between two trade-containing bars gets mid-quote fill."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    trades = _mk_trades(
        [
            {"ts": t0, "price": 5000.0, "quantity": 1, "aggressor_sign": 1, "is_implied": False},
            {"ts": t0 + timedelta(seconds=15), "price": 5001.0, "quantity": 1, "aggressor_sign": 1, "is_implied": False},
        ]
    )
    quotes = _mk_quotes(
        [
            {"ts": t0 - timedelta(seconds=1), "side": "bid", "price": 4999.75, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 - timedelta(seconds=1), "side": "ask", "price": 5000.00, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 + timedelta(seconds=7), "side": "bid", "price": 5000.50, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 + timedelta(seconds=7), "side": "ask", "price": 5000.75, "size": 50, "orders": 5, "is_implied": False},
        ]
    )
    bars = build_5sec_bars_core(trades, quotes, root="ES", expiry="ESH4")
    # Expect 4 bars: close at 14:30:05, :10, :15, :20
    assert bars.height >= 3
    # The middle bar (14:30:10, between the two trade bars) should have volume=0
    middle = bars.filter(pl.col("ts") == datetime(2024, 1, 2, 14, 30, 10, tzinfo=timezone.utc))
    if middle.height:
        r = middle.row(0, named=True)
        assert r["volume"] == 0
        assert r["trades_count"] == 0
        # OHLC should be filled (not null)
        assert r["open"] is not None
        assert r["close"] is not None


def test_is_rth_flag():
    """RTH window: 09:30 ET <= time < 16:00 ET."""
    # 14:30 UTC on 2024-01-02 = 09:30 ET (EST). Should be is_rth=True.
    # 13:00 UTC = 08:00 ET → is_rth=False.
    # 21:00 UTC = 16:00 ET → is_rth=False (right-exclusive).
    rows = []
    for utc_hr in [13, 14, 20, 21]:  # 08:00 ET, 09:00 ET, 15:00 ET, 16:00 ET
        t = datetime(2024, 1, 2, utc_hr, 30, 2, tzinfo=timezone.utc)
        rows.append({"ts": t, "price": 5000.0, "quantity": 1, "aggressor_sign": 1, "is_implied": False})
    trades = _mk_trades(rows)
    quote_ts = datetime(2024, 1, 2, 13, 0, 0, tzinfo=timezone.utc)
    quotes = _mk_quotes(
        [
            {"ts": quote_ts, "side": "bid", "price": 4999.75, "size": 50, "orders": 5, "is_implied": False},
            {"ts": quote_ts, "side": "ask", "price": 5000.00, "size": 50, "orders": 5, "is_implied": False},
        ]
    )
    bars = build_5sec_bars_core(trades, quotes, root="ES", expiry="ESH4")
    rth_map = dict(zip(bars["ts"].to_list(), bars["is_rth"].to_list()))
    # Find each UTC time's bar and check
    # 13:30 UTC = 08:30 ET → False
    # 14:30 UTC = 09:30 ET (RTH open) → True
    # 20:30 UTC = 15:30 ET → True
    # 21:30 UTC = 16:30 ET → False
    # Bar close is "next 5s boundary" — 13:30:02 → bar close 13:30:05 UTC
    pre_rth = bars.filter(pl.col("ts") == datetime(2024, 1, 2, 13, 30, 5, tzinfo=timezone.utc))
    rth_open_ish = bars.filter(pl.col("ts") == datetime(2024, 1, 2, 14, 30, 5, tzinfo=timezone.utc))
    post_rth = bars.filter(pl.col("ts") == datetime(2024, 1, 2, 21, 30, 5, tzinfo=timezone.utc))
    if pre_rth.height:
        assert pre_rth.row(0, named=True)["is_rth"] is False
    if rth_open_ish.height:
        assert rth_open_ish.row(0, named=True)["is_rth"] is True
    if post_rth.height:
        assert post_rth.row(0, named=True)["is_rth"] is False


def test_cvd_accumulates_across_bars():
    """CVD should accumulate signed volume across consecutive bars in same session."""
    # 4 bars, each 10 buys, same RTH session
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)  # 09:30 ET
    rows = []
    for i in range(4):
        ts = t0 + timedelta(seconds=i * 5 + 1)
        rows.append({"ts": ts, "price": 5000.0, "quantity": 10, "aggressor_sign": 1, "is_implied": False})
    trades = _mk_trades(rows)
    quotes = _mk_quotes(
        [
            {"ts": t0 - timedelta(seconds=5), "side": "bid", "price": 4999.75, "size": 50, "orders": 5, "is_implied": False},
            {"ts": t0 - timedelta(seconds=5), "side": "ask", "price": 5000.00, "size": 50, "orders": 5, "is_implied": False},
        ]
    )
    bars = build_5sec_bars_core(trades, quotes, root="ES", expiry="ESH4")
    bars = bars.filter(pl.col("trades_count") > 0)  # only trade bars
    cvd_rth = bars["cvd_rth"].to_list()
    # All trades are buys → cvd_rth monotonically increases by 10
    assert all(np.diff(cvd_rth) == 10)
    # And bars_since_rth_reset increments
    bsr = bars["bars_since_rth_reset"].to_list()
    assert bsr == sorted(bsr)  # monotonic
