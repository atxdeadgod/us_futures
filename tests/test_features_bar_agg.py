"""Tests for src/features/bar_agg.py."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import bar_agg


def test_order_flow_imbalance_signs():
    df = pl.DataFrame({"b": [80, 50, 20], "s": [20, 50, 80]})
    df = df.with_columns(
        bar_agg.order_flow_imbalance(pl.col("b"), pl.col("s")).alias("ofi")
    )
    # 80/20 → 0.6 (buys-dominant); 50/50 → 0; 20/80 → -0.6
    vals = df["ofi"].to_list()
    assert abs(vals[0] - 0.6) < 1e-6
    assert abs(vals[1]) < 1e-6
    assert abs(vals[2] + 0.6) < 1e-6


def test_aggressor_side_ratio():
    df = pl.DataFrame({"b": [80, 50, 20], "s": [20, 50, 80]})
    df = df.with_columns(
        bar_agg.aggressor_side_ratio(pl.col("b"), pl.col("s")).alias("ratio")
    )
    # 80/100 = 0.8; 50/100 = 0.5; 20/100 = 0.2
    assert [round(v, 2) for v in df["ratio"].to_list()] == [0.8, 0.5, 0.2]


def test_net_aggressor_signed():
    df = pl.DataFrame({"b": [80, 50, 20], "s": [20, 50, 80]})
    df = df.with_columns(
        bar_agg.net_aggressor_volume(pl.col("b"), pl.col("s")).alias("net")
    )
    assert df["net"].to_list() == [60, 0, -60]


def test_large_trade_volume_share():
    df = pl.DataFrame({"large": [20, 0, 100], "vol": [100, 100, 100]})
    df = df.with_columns(
        bar_agg.large_trade_volume_share(pl.col("large"), pl.col("vol")).alias("lts")
    )
    vals = df["lts"].to_list()
    assert abs(vals[0] - 0.2) < 1e-6
    assert abs(vals[1]) < 1e-6
    assert abs(vals[2] - 1.0) < 1e-6


def test_bid_ask_depth_ratio():
    """Equal depth → ratio = 1; bid-heavy → ratio > 1."""
    cols = {}
    for k in range(1, 11):
        cols[f"bid_sz_L{k}"] = [100, 100, 100]
        cols[f"ask_sz_L{k}"] = [100, 100, 100]
    df = pl.DataFrame(cols)
    df = df.with_columns(bar_agg.bid_ask_depth_ratio(levels=5).alias("r"))
    assert all(abs(v - 1.0) < 1e-6 for v in df["r"].to_list())

    cols2 = {}
    for k in range(1, 11):
        cols2[f"bid_sz_L{k}"] = [200, 200, 200]
        cols2[f"ask_sz_L{k}"] = [100, 100, 100]
    df2 = pl.DataFrame(cols2)
    df2 = df2.with_columns(bar_agg.bid_ask_depth_ratio(levels=5).alias("r"))
    assert all(abs(v - 2.0) < 1e-6 for v in df2["r"].to_list())


def test_side_weighted_spread_symmetric_zero():
    """Equal bid/ask sizes → imbalance = 0 → side-weighted spread = 0."""
    df = pl.DataFrame({"b1": [50, 50], "a1": [50, 50], "sp": [0.25, 0.25]})
    df = df.with_columns(
        bar_agg.side_weighted_spread_topbook(pl.col("b1"), pl.col("a1"), pl.col("sp")).alias("sws")
    )
    assert all(abs(v) < 1e-6 for v in df["sws"].to_list())


def test_average_trade_size():
    df = pl.DataFrame({"v": [100, 100, 50], "n": [10, 1, 5]})
    df = df.with_columns(
        bar_agg.average_trade_size(pl.col("v"), pl.col("n")).alias("ats")
    )
    assert [round(v, 2) for v in df["ats"].to_list()] == [10.0, 100.0, 10.0]


def test_rolling_volume_sum():
    df = pl.DataFrame({"v": [10, 20, 30, 40, 50]})
    df = df.with_columns(bar_agg.rolling_volume_sum(pl.col("v"), window=3).alias("rs"))
    # First 2 rows null (rolling warmup); 3rd = 60, 4th = 90, 5th = 120
    assert df["rs"].to_list()[2:] == [60, 90, 120]
