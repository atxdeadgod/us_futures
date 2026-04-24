"""Deep Order Flow Imbalance — multi-level signed book flow (T1.31).

Reference:
  Cont, Kukanov, Stoikov (2014) "The price impact of order book events" — L1 formula
  Kolm, Turiel, Westray (2023) "Deep Order Flow Imbalance" — L1-L10 extension

Per-level OFI at level k between consecutive book snapshots (rows) in the
dataframe:

  OFI_bid_k(t) =
    + bid_sz_k(t)                        if bid_px_k(t) >  bid_px_k(t-1)  # price improved
    + bid_sz_k(t) − bid_sz_k(t-1)        if bid_px_k(t) == bid_px_k(t-1)  # same price, size ΔQ
    − bid_sz_k(t-1)                      if bid_px_k(t) <  bid_px_k(t-1)  # price worsened

  OFI_ask_k(t) =   (symmetric with sign flipped; ask "new low" = selling pressure)
    − ask_sz_k(t)                        if ask_px_k(t) <  ask_px_k(t-1)
    − (ask_sz_k(t) − ask_sz_k(t-1))      if ask_px_k(t) == ask_px_k(t-1)
    + ask_sz_k(t-1)                      if ask_px_k(t) >  ask_px_k(t-1)

  OFI_k(t) = OFI_bid_k + OFI_ask_k          (positive = net buy pressure at level k)

Deep OFI aggregates across levels 1..max_depth with decay weighting:

  deep_ofi(t) = Σ_{k=1..D}  exp(−λ·(k−1)) · OFI_k(t)

λ=0 → uniform weighting; λ=1 → strong decay (L1 dominates).

Cross-market variant: pass a symbol prefix (e.g. "NQ") and the functions pull
from NQ_bid_px_L1 etc. on a joined wide panel — this is T3.08/T3.09.
"""
from __future__ import annotations

import math

import polars as pl


def _col(prefix: str, name: str) -> pl.Expr:
    """Column reference with optional symbol prefix."""
    return pl.col(f"{prefix}_{name}" if prefix else name)


def ofi_at_level(k: int, prefix: str = "") -> pl.Expr:
    """Per-level OFI contribution (bid + ask combined), positive = buy pressure.

    Uses polars .shift(1) to reference the previous row (previous bar-close snapshot).
    First row produces NaN (no prior snapshot).
    """
    bid_px = _col(prefix, f"bid_px_L{k}")
    bid_sz = _col(prefix, f"bid_sz_L{k}")
    ask_px = _col(prefix, f"ask_px_L{k}")
    ask_sz = _col(prefix, f"ask_sz_L{k}")
    bid_px_prev = bid_px.shift(1)
    bid_sz_prev = bid_sz.shift(1)
    ask_px_prev = ask_px.shift(1)
    ask_sz_prev = ask_sz.shift(1)

    # Bid-side OFI (positive on new-high bid or growing size)
    bid_ofi = (
        pl.when(bid_px > bid_px_prev)
        .then(bid_sz.cast(pl.Float64))
        .when(bid_px == bid_px_prev)
        .then((bid_sz - bid_sz_prev).cast(pl.Float64))
        .otherwise(-bid_sz_prev.cast(pl.Float64))
    )

    # Ask-side OFI (positive on growing ask = sellers pulling out; negative on
    # new-low ask = sellers aggressing inward)
    ask_ofi = (
        pl.when(ask_px < ask_px_prev)
        .then(-ask_sz.cast(pl.Float64))
        .when(ask_px == ask_px_prev)
        .then(-(ask_sz - ask_sz_prev).cast(pl.Float64))
        .otherwise(ask_sz_prev.cast(pl.Float64))
    )

    return bid_ofi + ask_ofi


def deep_ofi(
    max_depth: int = 10, decay: float = 0.0, prefix: str = "", normalize: bool = False
) -> pl.Expr:
    """Aggregated multi-level OFI.

    decay=0.0  → uniform weights (pure sum)
    decay=1.0  → exp decay: L1 weight=1, L2=0.37, L3=0.14, ...
    normalize=True → divide by Σ weights so result is a weighted AVERAGE rather than sum
    """
    assert max_depth >= 1, "max_depth must be >= 1"
    parts: list[pl.Expr] = []
    weights: list[float] = []
    for k in range(1, max_depth + 1):
        w = math.exp(-decay * (k - 1))
        weights.append(w)
        parts.append(w * ofi_at_level(k, prefix=prefix))
    total: pl.Expr = parts[0]
    for p in parts[1:]:
        total = total + p
    if normalize:
        total = total / sum(weights)
    return total


def ofi_per_level_columns(
    max_depth: int = 10, prefix: str = ""
) -> list[tuple[str, pl.Expr]]:
    """Returns a list of (column_name, expression) pairs — one per level 1..max_depth.

    Useful for emitting the per-level OFI columns from FEATURES.md schema (ofi_L1..L10).
    Caller uses:
        df.with_columns([expr.alias(name) for name, expr in pairs])
    """
    out = []
    for k in range(1, max_depth + 1):
        name = f"ofi_L{k}" if not prefix else f"{prefix}_ofi_L{k}"
        out.append((name, ofi_at_level(k, prefix=prefix)))
    return out


# ---------------------------------------------------------------------------
# Cross-market DeepOFI helpers (T3.08, T3.09)
# ---------------------------------------------------------------------------

def cross_market_deep_ofi(
    leader_prefix: str,
    max_depth: int = 10,
    decay: float = 0.0,
) -> pl.Expr:
    """T3.08/T3.09: DeepOFI computed on the LEADER's book (e.g. NQ, ZN) for use
    as a cross-asset predictor of the follower (ES).

    Caller assembles a wide panel with leader's L1-L10 book columns prefixed
    (NQ_bid_px_L1, etc.) and ES's (unprefixed) columns at aligned timestamps.
    """
    return deep_ofi(max_depth=max_depth, decay=decay, prefix=leader_prefix)
