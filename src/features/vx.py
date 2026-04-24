"""VX futures regime features — T5.01-T5.07.

Computed from our VX (VIX futures) L2 bar schema. Prefixes expected:
    VX1_*, VX2_*, VX3_*   (front, next, third-month VX contracts)
with columns bid_px_L1, ask_px_L1, bid_sz_L1, ask_sz_L1, ... for each.

Since we've built a generic bar builder + deep_ofi + l2 modules, VX features
reuse those with a prefix.

Functions:
    T5.01 vx_mid(prefix)                    (bid + ask) / 2
    T5.02 vx_zscore(mid_col, window)        (mid - rolling_mean) / rolling_std
    T5.03 vx_calendar_spread                VX1_mid - VX2_mid
    T5.04 vx_calendar_ratio                 VX1_mid / VX2_mid
    T5.05 vx_ofi_weighted(prefix, ...)      reuses deep_ofi.deep_ofi
    T5.06 vx_spread_zscore(prefix, window)  reuses l2.spread_zscore with prefix
    T5.07 vx_term_curvature                 VX3_mid − 2·VX2_mid + VX1_mid
"""
from __future__ import annotations

import polars as pl

from . import deep_ofi as _deep_ofi

EPS = 1e-9


# ---------------------------------------------------------------------------
# T5.01  VX1 mid price
# ---------------------------------------------------------------------------

def vx_mid(prefix: str = "VX1") -> pl.Expr:
    """(bid + ask) / 2 for the given VX contract prefix (VX1, VX2, VX3)."""
    return (pl.col(f"{prefix}_bid_px_L1") + pl.col(f"{prefix}_ask_px_L1")) / 2.0


# ---------------------------------------------------------------------------
# T5.02  VX1 rolling z-score
# ---------------------------------------------------------------------------

def vx_zscore(mid_col: pl.Expr, window: int = 20) -> pl.Expr:
    """T5.02: (mid - rolling_mean) / rolling_std over `window` bars.

    At 5-sec bars with window=20 → ~100 sec lookback; at 15-min bars with
    window=20*21 ≈ 420 → ~6 hours. Caller picks the right window for their
    target horizon.
    """
    rmean = mid_col.rolling_mean(window_size=window)
    rstd = mid_col.rolling_std(window_size=window)
    return (mid_col - rmean) / (rstd + EPS)


# ---------------------------------------------------------------------------
# T5.03 / T5.04  Calendar spread + ratio
# ---------------------------------------------------------------------------

def vx_calendar_spread(
    front_prefix: str = "VX1", back_prefix: str = "VX2"
) -> pl.Expr:
    """T5.03: front VX mid − back VX mid. Positive = backwardation (rare, risk-off)."""
    return vx_mid(front_prefix) - vx_mid(back_prefix)


def vx_calendar_ratio(
    front_prefix: str = "VX1", back_prefix: str = "VX2"
) -> pl.Expr:
    """T5.04: front / back. Ratio > 1 = backwardation; < 1 = contango (normal)."""
    return vx_mid(front_prefix) / (vx_mid(back_prefix) + EPS)


# ---------------------------------------------------------------------------
# T5.05  VX OFI (deep multi-level on VX book)
# ---------------------------------------------------------------------------

def vx_ofi_weighted(
    prefix: str = "VX1",
    max_depth: int = 10,
    decay: float = 0.0,
) -> pl.Expr:
    """T5.05: applies deep_ofi.deep_ofi to the VX1 book.

    Positive flow into volatility = bullish-vol pressure (risk-off regime).
    """
    return _deep_ofi.deep_ofi(
        max_depth=max_depth, decay=decay, prefix=prefix
    )


# ---------------------------------------------------------------------------
# T5.06  VX spread z-score  (reuses logic of l2.spread_zscore but with prefix)
# ---------------------------------------------------------------------------

def vx_spread_zscore(
    prefix: str = "VX1", depth: int = 1, window: int = 60
) -> pl.Expr:
    """T5.06: Rolling z-score of the at-depth-k spread for the VX contract.

    Reuses the same formula as l2.spread_zscore but with a prefix-aware column
    reference.
    """
    sp = pl.col(f"{prefix}_ask_px_L{depth}") - pl.col(f"{prefix}_bid_px_L{depth}")
    return (sp - sp.rolling_mean(window_size=window)) / (sp.rolling_std(window_size=window) + EPS)


# ---------------------------------------------------------------------------
# T5.07  VX term-structure curvature
# ---------------------------------------------------------------------------

def vx_term_curvature(
    front: str = "VX1", mid: str = "VX2", back: str = "VX3"
) -> pl.Expr:
    """T5.07: second difference across VX1/VX2/VX3 mids.

    curvature = VX3_mid − 2·VX2_mid + VX1_mid
    Positive = curve is concave UP (bowl-shaped, backwardation-like in middle)
    Negative = concave DOWN (hump, typical vol-term-structure shape)
    """
    return vx_mid(back) - 2 * vx_mid(mid) + vx_mid(front)
