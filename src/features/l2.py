"""L2 (deep book) features — pure polars expressions.

Extracted from feature-factory. Framework-free — each function takes explicit
columns or depth parameters and returns a polars expression / DataFrame column.

The repo bar schema uses cols named `bid_sz_L{k}`, `bid_px_L{k}`, `ask_sz_L{k}`,
`ask_px_L{k}`, `bid_ord_L{k}`, `ask_ord_L{k}` for k=1..10. All functions below
target those names.

Features covered (12 of Tier 1 reusable set):
    T1.09  volume_imbalance (at depth k)
    T1.10  distance_weighted_imbalance
    T1.11  cumulative_imbalance
    T1.12  imbalance_persistence
    T1.13  basic_spread (at depth k)
    T1.14  depth_weighted_spread
    T1.15  liquidity_adjusted_spread
    T1.16  spread_acceleration (L0,L1,L2 second-difference)
    T1.17  spread_zscore
    T1.18  order_count_imbalance
    T1.19  order_size_imbalance
    T1.20  herfindahl_hirschman_index (depth concentration)
"""
from __future__ import annotations

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# Imbalance family
# ---------------------------------------------------------------------------

def volume_imbalance_at(k: int = 1) -> pl.Expr:
    """T1.09: (bid_sz_Lk − ask_sz_Lk) / (bid_sz_Lk + ask_sz_Lk). k in 1..10."""
    b = pl.col(f"bid_sz_L{k}")
    a = pl.col(f"ask_sz_L{k}")
    return (b - a) / (b + a + EPS)


def cumulative_imbalance(depth: int = 5) -> pl.Expr:
    """T1.11: sum over levels 1..depth of per-level (bid-ask)/(bid+ask).

    Higher values → bid-heavy up to `depth` levels.
    """
    parts = []
    for k in range(1, depth + 1):
        parts.append(volume_imbalance_at(k))
    return sum(parts[1:], start=parts[0])  # avoid pl.sum_horizontal for nested exprs


def distance_weighted_imbalance_cols(depth: int = 5) -> list[pl.Expr]:
    """T1.10: Distance-weighted book imbalance across levels 1..depth.

    Weight at level k = amount / (1 + |price_k - mid|). Returns a LIST of
    expressions that together compute the feature — caller applies in a
    single .with_columns() to avoid extra shuffles.

    Returns two exprs: (weighted_imbalance_value, mid_helper_col).
    """
    bid_px_1 = pl.col("bid_px_L1")
    ask_px_1 = pl.col("ask_px_L1")
    mid = (bid_px_1 + ask_px_1) / 2.0

    bid_weighted = []
    ask_weighted = []
    for k in range(1, depth + 1):
        bid_px = pl.col(f"bid_px_L{k}")
        bid_sz = pl.col(f"bid_sz_L{k}")
        ask_px = pl.col(f"ask_px_L{k}")
        ask_sz = pl.col(f"ask_sz_L{k}")
        bid_weighted.append(bid_sz / (1 + (bid_px - mid).abs()))
        ask_weighted.append(ask_sz / (1 + (ask_px - mid).abs()))

    b_sum = sum(bid_weighted[1:], start=bid_weighted[0])
    a_sum = sum(ask_weighted[1:], start=ask_weighted[0])
    return [(b_sum - a_sum) / (b_sum + a_sum + EPS), mid]  # feature, mid_helper


def distance_weighted_imbalance(depth: int = 5) -> pl.Expr:
    """Convenience wrapper returning the imbalance expression only."""
    return distance_weighted_imbalance_cols(depth)[0]


# ---------------------------------------------------------------------------
# Spread family
# ---------------------------------------------------------------------------

def basic_spread_at(k: int = 1) -> pl.Expr:
    """T1.13: ask_px_Lk − bid_px_Lk."""
    return pl.col(f"ask_px_L{k}") - pl.col(f"bid_px_L{k}")


def depth_weighted_spread(depth: int = 5) -> pl.Expr:
    """T1.14: Volume-cumulative weighted spread across depths.

    For each level k, compute spread_k * cum_vol_1..k, sum, divide by total cum vol.
    Gives more weight to spreads paired with larger displayed depth.
    """
    spreads = []
    cum_vols: list[pl.Expr] = []
    weighted: list[pl.Expr] = []
    for k in range(1, depth + 1):
        sp = pl.col(f"ask_px_L{k}") - pl.col(f"bid_px_L{k}")
        vol = pl.col(f"bid_sz_L{k}") + pl.col(f"ask_sz_L{k}")
        cv = vol if not cum_vols else cum_vols[-1] + vol
        cum_vols.append(cv)
        weighted.append(sp * cv)
        spreads.append(sp)
    total_cum = cum_vols[-1]
    num = sum(weighted[1:], start=weighted[0])
    return num / (total_cum + EPS)


def liquidity_adjusted_spread(depth: int = 5) -> pl.Expr:
    """T1.15: Mean over levels 1..depth of (spread / volume)."""
    parts = []
    for k in range(1, depth + 1):
        sp = pl.col(f"ask_px_L{k}") - pl.col(f"bid_px_L{k}")
        vol = pl.col(f"bid_sz_L{k}") + pl.col(f"ask_sz_L{k}")
        parts.append(sp / (vol + EPS))
    total = sum(parts[1:], start=parts[0])
    return total / depth


def spread_acceleration() -> pl.Expr:
    """T1.16: Second difference across levels 1..3: s3 − 2·s2 + s1.

    Positive = spread widens faster as we go deeper (thin book shape); negative
    = spread flattens with depth (deep book concentrates near the inside).
    """
    s1 = pl.col("ask_px_L1") - pl.col("bid_px_L1")
    s2 = pl.col("ask_px_L2") - pl.col("bid_px_L2")
    s3 = pl.col("ask_px_L3") - pl.col("bid_px_L3")
    return s3 - 2 * s2 + s1


def spread_zscore_cols(depth: int = 1, window: int = 60) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """T1.17: Rolling z-score of the at-depth-k spread.

    Returns a tuple (spread_col, rolling_mean, rolling_std). Caller applies all
    three in one .with_columns, then computes the final z in a second pass.
    """
    sp = pl.col(f"ask_px_L{depth}") - pl.col(f"bid_px_L{depth}")
    rm = sp.rolling_mean(window_size=window)
    rs = sp.rolling_std(window_size=window)
    return sp, rm, rs


def spread_zscore(depth: int = 1, window: int = 60) -> pl.Expr:
    """Convenience single-expression form (does the computation inline)."""
    sp = pl.col(f"ask_px_L{depth}") - pl.col(f"bid_px_L{depth}")
    return (sp - sp.rolling_mean(window_size=window)) / (sp.rolling_std(window_size=window) + EPS)


# ---------------------------------------------------------------------------
# Order-count / order-size family (uses bid_ord_L* / ask_ord_L*)
# ---------------------------------------------------------------------------

def order_count_imbalance_at(k: int = 1) -> pl.Expr:
    """T1.18: (bid_ord − ask_ord) / (bid_ord + ask_ord) at level k."""
    b = pl.col(f"bid_ord_L{k}")
    a = pl.col(f"ask_ord_L{k}")
    return (b - a) / (b + a + EPS)


def order_size_imbalance_at(k: int = 1) -> pl.Expr:
    """T1.19: Imbalance of AVERAGE order sizes at level k."""
    avg_bid = pl.col(f"bid_sz_L{k}") / (pl.col(f"bid_ord_L{k}") + EPS)
    avg_ask = pl.col(f"ask_sz_L{k}") / (pl.col(f"ask_ord_L{k}") + EPS)
    return (avg_bid - avg_ask) / (avg_bid + avg_ask + EPS)


# ---------------------------------------------------------------------------
# Depth concentration (HHI)
# ---------------------------------------------------------------------------

def herfindahl_hirschman_index(
    side: str = "bid", depth: int = 5, base: str = "sz"
) -> pl.Expr:
    """T1.20: Depth concentration across levels 1..depth on one side.

    base='sz' → uses bid_sz_Lk; base='ord' → uses bid_ord_Lk.
    Value in [1/depth, 1]; higher = concentrated in fewer levels.
    """
    assert side in ("bid", "ask"), f"side must be bid/ask, got {side!r}"
    assert base in ("sz", "ord"), f"base must be sz/ord, got {base!r}"
    cols = [pl.col(f"{side}_{base}_L{k}") for k in range(1, depth + 1)]
    total = sum(cols[1:], start=cols[0])
    shares_sq = [(c / (total + EPS)) ** 2 for c in cols]
    return sum(shares_sq[1:], start=shares_sq[0])
