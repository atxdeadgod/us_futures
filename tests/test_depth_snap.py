"""Unit tests for src/data/depth_snap.py."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.depth_snap import attach_book_snapshot, split_sides


def _mk_depth(rows):
    """rows: list of dicts with ts, Side ('B'/'S'), Flags (int), + L{k}Price/Size/Orders."""
    schema: dict[str, pl.DataType] = {
        "ts": pl.Datetime("ns", "UTC"),
        "Side": pl.Utf8,
        "Flags": pl.Int64,
    }
    for k in range(1, 11):
        schema[f"L{k}Price"] = pl.Float64
        schema[f"L{k}Size"] = pl.Int64
        schema[f"L{k}Orders"] = pl.Int64
    df = pl.DataFrame(rows, schema=schema)
    return df


def _mk_bars(ts_list):
    return pl.DataFrame({"ts": ts_list}).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


def _bid_row(ts, **kwargs):
    r = {"ts": ts, "Side": "B", "Flags": 0}
    for k in range(1, 11):
        r[f"L{k}Price"] = kwargs.get(f"p{k}", 0.0)
        r[f"L{k}Size"] = kwargs.get(f"s{k}", 0)
        r[f"L{k}Orders"] = kwargs.get(f"o{k}", 0)
    return r


def _ask_row(ts, **kwargs):
    r = {"ts": ts, "Side": "S", "Flags": 0}
    for k in range(1, 11):
        r[f"L{k}Price"] = kwargs.get(f"p{k}", 0.0)
        r[f"L{k}Size"] = kwargs.get(f"s{k}", 0)
        r[f"L{k}Orders"] = kwargs.get(f"o{k}", 0)
    return r


def test_split_sides_regular_only():
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    depth = _mk_depth(
        [
            _bid_row(t0, p1=5000.0, s1=10, o1=2),
            _ask_row(t0, p1=5000.25, s1=12, o1=3),
            {**_bid_row(t0 + timedelta(seconds=1), p1=5000.0, s1=15, o1=4), "Flags": 1},  # implied
        ]
    )
    bid, ask = split_sides(depth, only_regular=True)
    assert bid.height == 1, "implied bid should be filtered"
    assert ask.height == 1
    assert bid.columns[:4] == ["ts", "bid_px_L1", "bid_sz_L1", "bid_ord_L1"]
    assert ask.columns[:4] == ["ts", "ask_px_L1", "ask_sz_L1", "ask_ord_L1"]
    assert bid.row(0, named=True)["bid_px_L1"] == 5000.0
    assert ask.row(0, named=True)["ask_px_L1"] == 5000.25


def test_split_sides_keeps_implied_when_requested():
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    depth = _mk_depth(
        [
            _bid_row(t0, p1=5000.0, s1=10, o1=2),
            {**_bid_row(t0 + timedelta(seconds=1), p1=5000.0, s1=15, o1=4), "Flags": 1},
        ]
    )
    bid, _ask = split_sides(depth, only_regular=False)
    assert bid.height == 2, "both regular + implied should be kept"


def test_attach_snapshot_uses_most_recent_before_bar():
    t_snap = datetime(2024, 1, 2, 14, 30, 3, tzinfo=timezone.utc)
    t_bar = datetime(2024, 1, 2, 14, 30, 5, tzinfo=timezone.utc)  # bar closes here
    depth = _mk_depth(
        [
            _bid_row(t_snap, p1=5000.0, s1=10, o1=2),
            _ask_row(t_snap, p1=5000.25, s1=12, o1=3),
        ]
    )
    bars = _mk_bars([t_bar])
    out = attach_book_snapshot(bars, depth)
    row = out.row(0, named=True)
    assert row["bid_px_L1"] == 5000.0
    assert row["bid_sz_L1"] == 10
    assert row["ask_px_L1"] == 5000.25
    # book_ts_close should equal t_snap (most recent of bid/ask, both at t_snap)
    assert row["book_ts_close"] == t_snap


def test_attach_snapshot_no_lookahead_at_exact_bar_close():
    """A depth event at exactly bar_close must NOT contaminate that bar."""
    t_bar = datetime(2024, 1, 2, 14, 30, 5, tzinfo=timezone.utc)
    t_earlier = t_bar - timedelta(seconds=1)
    depth = _mk_depth(
        [
            _bid_row(t_earlier, p1=4999.0, s1=5, o1=1),  # earlier snap — should be used
            _ask_row(t_earlier, p1=4999.25, s1=5, o1=1),
            _bid_row(t_bar, p1=5000.0, s1=99, o1=99),  # at exact bar close — must NOT be used
            _ask_row(t_bar, p1=5000.25, s1=99, o1=99),
        ]
    )
    bars = _mk_bars([t_bar])
    out = attach_book_snapshot(bars, depth)
    row = out.row(0, named=True)
    # Strict < rule: the exact-bar-close depth event is excluded; use t_earlier snap
    assert row["bid_px_L1"] == 4999.0
    assert row["bid_sz_L1"] == 5
    assert row["ask_px_L1"] == 4999.25
    assert row["book_ts_close"] == t_earlier


def test_attach_snapshot_bar_before_any_depth_is_null():
    t_bar = datetime(2024, 1, 2, 14, 30, 5, tzinfo=timezone.utc)
    t_later = datetime(2024, 1, 2, 14, 30, 10, tzinfo=timezone.utc)
    depth = _mk_depth(
        [
            _bid_row(t_later, p1=5000.0, s1=10, o1=2),
            _ask_row(t_later, p1=5000.25, s1=10, o1=2),
        ]
    )
    bars = _mk_bars([t_bar])
    out = attach_book_snapshot(bars, depth)
    row = out.row(0, named=True)
    assert row["bid_px_L1"] is None
    assert row["ask_px_L1"] is None
    assert row["book_ts_close"] is None


def test_attach_snapshot_forward_fills_between_events():
    """Between two depth events, the latter bars keep the more recent event until a new one."""
    t0 = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    depth = _mk_depth(
        [
            _bid_row(t0 + timedelta(seconds=1), p1=5000.0, s1=10, o1=2),
            _ask_row(t0 + timedelta(seconds=1), p1=5000.25, s1=10, o1=2),
            _bid_row(t0 + timedelta(seconds=6), p1=5000.5, s1=20, o1=4),  # new bid at t+6
            # no new ask until later
        ]
    )
    bars = _mk_bars(
        [
            t0 + timedelta(seconds=5),   # bar 1 close → uses t+1 snap for both sides
            t0 + timedelta(seconds=10),  # bar 2 close → uses t+6 bid, t+1 ask
        ]
    )
    out = attach_book_snapshot(bars, depth)
    r1 = out.row(0, named=True)
    r2 = out.row(1, named=True)
    assert r1["bid_px_L1"] == 5000.0
    assert r1["ask_px_L1"] == 5000.25
    assert r2["bid_px_L1"] == 5000.5   # updated
    assert r2["ask_px_L1"] == 5000.25  # unchanged (no new ask)
