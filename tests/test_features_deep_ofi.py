"""Tests for src/features/deep_ofi.py — multi-level OFI.

Each test builds a two-row (or few-row) book snapshot and verifies the OFI
value against a hand-computed expectation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import math
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import deep_ofi


def _mk_book_rows(rows, max_depth=3, prefix=""):
    """rows: list of dicts like {'bid_px_L1': 100, 'bid_sz_L1': 10, 'ask_px_L1': 101, 'ask_sz_L1': 15, ...}.
    Fills missing levels with zeros.
    """
    cols = {}
    for r in rows:
        for k in range(1, max_depth + 1):
            for field in (f"bid_px_L{k}", f"bid_sz_L{k}", f"ask_px_L{k}", f"ask_sz_L{k}"):
                name = f"{prefix}_{field}" if prefix else field
                cols.setdefault(name, []).append(r.get(field, 0.0))
    return pl.DataFrame(cols)


def test_ofi_first_row_null():
    """First row has no prior snapshot → OFI is null (from shift(1))."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 10, "ask_px_L1": 101, "ask_sz_L1": 10},
    ])
    df = df.with_columns(deep_ofi.ofi_at_level(1).alias("o"))
    assert df["o"][0] is None


def test_ofi_same_price_size_grows_positive():
    """Bid price stable, size grows 10→30 → OFI_bid = +20. Ask price stable, size grows 10→15 → OFI_ask = -5."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 10, "ask_px_L1": 101, "ask_sz_L1": 10},
        {"bid_px_L1": 100, "bid_sz_L1": 30, "ask_px_L1": 101, "ask_sz_L1": 15},
    ])
    df = df.with_columns(deep_ofi.ofi_at_level(1).alias("o"))
    # OFI_bid = 30 - 10 = 20; OFI_ask = -(15-10) = -5; total = 15
    assert df["o"][1] == 15.0


def test_ofi_bid_price_improves():
    """Bid price goes up: OFI_bid = +current_size (fresh interest)."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 10, "ask_px_L1": 101, "ask_sz_L1": 10},
        {"bid_px_L1": 101, "bid_sz_L1": 25, "ask_px_L1": 102, "ask_sz_L1": 10},  # ask px went up = cancelled
    ])
    df = df.with_columns(deep_ofi.ofi_at_level(1).alias("o"))
    # OFI_bid = +25 (bid price improved)
    # OFI_ask: ask_px > ask_px_prev → +ask_sz_prev = +10
    # total = 25 + 10 = 35
    assert df["o"][1] == 35.0


def test_ofi_bid_price_worsens():
    """Bid price down: OFI_bid = -prev_size (liquidity pulled)."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 40, "ask_px_L1": 101, "ask_sz_L1": 10},
        {"bid_px_L1": 99,  "bid_sz_L1": 20, "ask_px_L1": 100, "ask_sz_L1": 10},  # ask px down = sellers aggressing
    ])
    df = df.with_columns(deep_ofi.ofi_at_level(1).alias("o"))
    # OFI_bid = -40 (prior bid size pulled away)
    # OFI_ask = -10 (ask price improved → new selling pressure)
    # total = -50
    assert df["o"][1] == -50.0


def test_deep_ofi_uniform_weights():
    """With decay=0, deep_ofi = sum over levels of per-level OFI."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 10, "ask_px_L1": 101, "ask_sz_L1": 10,
         "bid_px_L2": 99,  "bid_sz_L2": 20, "ask_px_L2": 102, "ask_sz_L2": 20,
         "bid_px_L3": 98,  "bid_sz_L3": 30, "ask_px_L3": 103, "ask_sz_L3": 30},
        {"bid_px_L1": 100, "bid_sz_L1": 15, "ask_px_L1": 101, "ask_sz_L1": 10,
         "bid_px_L2": 99,  "bid_sz_L2": 20, "ask_px_L2": 102, "ask_sz_L2": 25,
         "bid_px_L3": 98,  "bid_sz_L3": 30, "ask_px_L3": 103, "ask_sz_L3": 30},
    ], max_depth=3)
    df = df.with_columns(deep_ofi.deep_ofi(max_depth=3, decay=0.0).alias("dofi"))
    # L1: bid +5, ask 0 → +5
    # L2: bid 0, ask -5 → -5
    # L3: bid 0, ask 0 → 0
    # sum = 0
    assert df["dofi"][1] == 0.0


def test_deep_ofi_decay_gives_different_weights():
    """With decay>0, L1 dominates."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 100, "ask_px_L1": 101, "ask_sz_L1": 100,
         "bid_px_L2": 99,  "bid_sz_L2": 100, "ask_px_L2": 102, "ask_sz_L2": 100},
        # L1 bid grows +50 (ofi +50), L2 ask grows +50 (ofi -50). Sum = 0 with uniform.
        {"bid_px_L1": 100, "bid_sz_L1": 150, "ask_px_L1": 101, "ask_sz_L1": 100,
         "bid_px_L2": 99,  "bid_sz_L2": 100, "ask_px_L2": 102, "ask_sz_L2": 150},
    ], max_depth=2)
    df = df.with_columns([
        deep_ofi.deep_ofi(max_depth=2, decay=0.0).alias("uniform"),
        deep_ofi.deep_ofi(max_depth=2, decay=1.0).alias("decayed"),
    ])
    # Uniform: +50 + (-50) = 0
    # Decayed: 1.0*50 + exp(-1)*(-50) = 50 - 18.4 ≈ 31.6 (positive, L1 dominates)
    assert df["uniform"][1] == 0.0
    assert df["decayed"][1] > 25.0
    assert df["decayed"][1] < 35.0


def test_per_level_columns_emit():
    """ofi_per_level_columns returns named-expr pairs; we can attach them all."""
    df = _mk_book_rows([
        {"bid_px_L1": 100, "bid_sz_L1": 10, "ask_px_L1": 101, "ask_sz_L1": 10,
         "bid_px_L2": 99,  "bid_sz_L2": 20, "ask_px_L2": 102, "ask_sz_L2": 20},
        {"bid_px_L1": 100, "bid_sz_L1": 25, "ask_px_L1": 101, "ask_sz_L1": 10,
         "bid_px_L2": 99,  "bid_sz_L2": 30, "ask_px_L2": 102, "ask_sz_L2": 20},
    ], max_depth=2)
    pairs = deep_ofi.ofi_per_level_columns(max_depth=2)
    df = df.with_columns([expr.alias(name) for name, expr in pairs])
    assert "ofi_L1" in df.columns
    assert "ofi_L2" in df.columns
    # L1: bid +15, ask 0 = +15
    # L2: bid +10, ask 0 = +10
    assert df["ofi_L1"][1] == 15.0
    assert df["ofi_L2"][1] == 10.0


def test_cross_market_deep_ofi_with_prefix():
    """Prefix-aware version works on NQ-prefixed columns."""
    df = _mk_book_rows([
        {"bid_px_L1": 17000, "bid_sz_L1": 100, "ask_px_L1": 17001, "ask_sz_L1": 100},
        {"bid_px_L1": 17000, "bid_sz_L1": 150, "ask_px_L1": 17001, "ask_sz_L1": 100},
    ], max_depth=1, prefix="NQ")
    df = df.with_columns(
        deep_ofi.cross_market_deep_ofi("NQ", max_depth=1, decay=0.0).alias("nq_ofi")
    )
    # L1: NQ bid +50, ask 0 → +50
    assert df["nq_ofi"][1] == 50.0
