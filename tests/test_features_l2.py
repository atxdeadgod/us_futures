"""Tests for src/features/l2.py — depth-book features on L1-L10 schema."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import l2


def _mk_book(n: int = 5, bid_sz_per_lvl=None, ask_sz_per_lvl=None, bid_ord_per_lvl=None, ask_ord_per_lvl=None, px_step: float = 0.25):
    """Build a synthetic 1-row book with n rows repeated.

    L1 bid = 5000.00, L1 ask = 5000.25. Each level k has bid_px_Lk = 5000 - (k-1)*step,
    ask_px_Lk = 5000.25 + (k-1)*step.
    """
    if bid_sz_per_lvl is None:
        bid_sz_per_lvl = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if ask_sz_per_lvl is None:
        ask_sz_per_lvl = bid_sz_per_lvl
    if bid_ord_per_lvl is None:
        bid_ord_per_lvl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if ask_ord_per_lvl is None:
        ask_ord_per_lvl = bid_ord_per_lvl
    cols = {}
    for k in range(1, 11):
        cols[f"bid_px_L{k}"] = [5000.0 - (k - 1) * px_step] * n
        cols[f"ask_px_L{k}"] = [5000.25 + (k - 1) * px_step] * n
        cols[f"bid_sz_L{k}"] = [bid_sz_per_lvl[k - 1]] * n
        cols[f"ask_sz_L{k}"] = [ask_sz_per_lvl[k - 1]] * n
        cols[f"bid_ord_L{k}"] = [bid_ord_per_lvl[k - 1]] * n
        cols[f"ask_ord_L{k}"] = [ask_ord_per_lvl[k - 1]] * n
    return pl.DataFrame(cols)


def test_volume_imbalance_at_balanced():
    df = _mk_book(bid_sz_per_lvl=[100] * 10, ask_sz_per_lvl=[100] * 10)
    df = df.with_columns(l2.volume_imbalance_at(1).alias("vi1"))
    assert all(abs(v) < 1e-6 for v in df["vi1"].to_list())


def test_volume_imbalance_at_bid_heavy():
    df = _mk_book(bid_sz_per_lvl=[90] * 10, ask_sz_per_lvl=[10] * 10)
    df = df.with_columns(l2.volume_imbalance_at(1).alias("vi1"))
    assert all(abs(v - 0.8) < 1e-6 for v in df["vi1"].to_list())  # (90-10)/100=0.8


def test_cumulative_imbalance_sums_levels():
    """With bid=2·ask at every level, per-level imbalance = 1/3. Sum over 3 levels = 1."""
    df = _mk_book(bid_sz_per_lvl=[20] * 10, ask_sz_per_lvl=[10] * 10)
    df = df.with_columns(l2.cumulative_imbalance(depth=3).alias("ci"))
    # Each level imbalance = (20-10)/30 = 0.333; sum over 3 = 1.0
    assert all(abs(v - 1.0) < 1e-6 for v in df["ci"].to_list())


def test_distance_weighted_imbalance_balanced():
    """Symmetric book → weighted imbalance ≈ 0."""
    df = _mk_book()  # default symmetric
    df = df.with_columns(l2.distance_weighted_imbalance(depth=5).alias("dwi"))
    assert all(abs(v) < 1e-6 for v in df["dwi"].to_list())


def test_basic_spread():
    df = _mk_book()  # L1 spread = 5000.25 - 5000.00 = 0.25
    df = df.with_columns(l2.basic_spread_at(1).alias("sp1"))
    assert all(abs(v - 0.25) < 1e-9 for v in df["sp1"].to_list())


def test_depth_weighted_spread_sane_range():
    """Weighted spread with constant 0.25 spread at every level lies in [0.25, N·0.25]
    range. For depth=3 with uniform-ish volumes, result ≈ 0.5 due to cum-vol weighting
    (inner levels appear in later cum_vols). Just verify it's between 0.25 and 0.75."""
    df = _mk_book(px_step=0.0, bid_sz_per_lvl=[10] * 10, ask_sz_per_lvl=[10] * 10)
    df = df.with_columns(l2.depth_weighted_spread(depth=3).alias("dws"))
    for v in df["dws"].to_list():
        assert 0.25 <= v <= 0.75


def test_liquidity_adjusted_spread_higher_vol_smaller():
    """Larger volumes → smaller liquidity-adjusted spread."""
    thin = _mk_book(bid_sz_per_lvl=[1] * 10, ask_sz_per_lvl=[1] * 10)
    thick = _mk_book(bid_sz_per_lvl=[100] * 10, ask_sz_per_lvl=[100] * 10)
    thin = thin.with_columns(l2.liquidity_adjusted_spread(depth=5).alias("las"))
    thick = thick.with_columns(l2.liquidity_adjusted_spread(depth=5).alias("las"))
    assert thin["las"][0] > thick["las"][0]


def test_spread_acceleration_uniform_is_zero():
    """If spreads are linear across L1/L2/L3, second difference = 0."""
    df = _mk_book(px_step=0.25)  # spread at Lk = 0.25 + 2*(k-1)*0.25
    # L1 sp = 0.25, L2 sp = 0.75, L3 sp = 1.25 — second diff = 1.25 - 2*0.75 + 0.25 = 0
    df = df.with_columns(l2.spread_acceleration().alias("sa"))
    assert all(abs(v) < 1e-9 for v in df["sa"].to_list())


def test_spread_zscore_rolling():
    """Constant spread over a long series → z-score becomes 0 after warmup."""
    df = _mk_book(n=100)
    df = df.with_columns(l2.spread_zscore(depth=1, window=20).alias("z"))
    valid = df["z"].drop_nulls().to_list()
    # All values should be 0 (constant spread → 0 std)... but actually std=0 → divide by EPS → blows up
    # With a constant series, the rolling_std is 0, spread - mean = 0 → 0/EPS = 0
    assert all(abs(v) < 1e-3 for v in valid)


def test_order_count_imbalance_at():
    df = _mk_book(bid_ord_per_lvl=[3] * 10, ask_ord_per_lvl=[1] * 10)
    df = df.with_columns(l2.order_count_imbalance_at(1).alias("oci"))
    # (3-1)/4 = 0.5
    assert all(abs(v - 0.5) < 1e-6 for v in df["oci"].to_list())


def test_hhi_fully_concentrated():
    """All volume at L1, zero elsewhere → HHI = 1."""
    bids = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    df = _mk_book(bid_sz_per_lvl=bids, ask_sz_per_lvl=bids)
    df = df.with_columns(l2.herfindahl_hirschman_index(side="bid", depth=5, base="sz").alias("hhi"))
    assert all(abs(v - 1.0) < 1e-6 for v in df["hhi"].to_list())


def test_hhi_uniform_equals_one_over_n():
    """Equal sizes across depth levels → HHI = 1/depth."""
    df = _mk_book(bid_sz_per_lvl=[10] * 10, ask_sz_per_lvl=[10] * 10)
    df = df.with_columns(l2.herfindahl_hirschman_index(side="bid", depth=5, base="sz").alias("hhi"))
    # 5 levels each with 20% share → HHI = 5 * 0.2² = 0.2
    assert all(abs(v - 0.2) < 1e-6 for v in df["hhi"].to_list())
