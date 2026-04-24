"""Pairwise L2-book cross features — pure polars expressions.

Assumes caller has assembled a JOINED frame with both symbols' L1/L2 columns
renamed with symbol prefixes (e.g. `ES_bid_px_L1`, `ES_ask_px_L1`, `NQ_bid_px_L1`,
`NQ_ask_px_L1` etc.) such that both symbols share the same row index.

Simpler-than-feature-factory: functions take symbol PREFIXES and return polars
expressions against the combined frame.

Covered:
    T3.11 cross_correlation      rolling Pearson(mid-return A, mid-return B)
    T3.11 depth_imbalance_diff   imb_A(depth) - imb_B(depth)
    T3.11 microprice_diff        microprice_A - microprice_B
    T3.11 ofi_correlation        rolling corr(top-of-book OFI_A, OFI_B)
    T3.11 price_lead_lag         corr product ret_A * ret_B.shift(lag)
    T3.11 pairs_spread_zscore    rolling z(mid_A - mid_B)
    T3.11 realized_vol_ratio     vol_A / vol_B
    T3.11 relative_quoted_spread relative spread_A - spread_B

Deferred (regression / state-space; V2):
    T3.11 hedge_ratio, rolling_beta, half_life_mean_reversion, information_share
"""
from __future__ import annotations

import polars as pl

EPS = 1e-9


def _mid(symbol_prefix: str) -> pl.Expr:
    """(bid_px_L1 + ask_px_L1) / 2 with `{symbol_prefix}_` column names."""
    return (pl.col(f"{symbol_prefix}_bid_px_L1") + pl.col(f"{symbol_prefix}_ask_px_L1")) / 2


def _log_return(symbol_prefix: str) -> pl.Expr:
    """Log-return of mid."""
    mid = _mid(symbol_prefix)
    return (mid / mid.shift(1)).log()


# ---------------------------------------------------------------------------
# Pairwise correlations
# ---------------------------------------------------------------------------

def cross_correlation(prefix_a: str, prefix_b: str, window: int) -> pl.Expr:
    """Rolling Pearson(mid_return_A, mid_return_B)."""
    ra = _log_return(prefix_a)
    rb = _log_return(prefix_b)
    mean_ra = ra.rolling_mean(window_size=window)
    mean_rb = rb.rolling_mean(window_size=window)
    cov = ((ra - mean_ra) * (rb - mean_rb)).rolling_mean(window_size=window)
    var_a = ((ra - mean_ra) ** 2).rolling_mean(window_size=window)
    var_b = ((rb - mean_rb) ** 2).rolling_mean(window_size=window)
    denom = (var_a * var_b).sqrt()
    return pl.when(denom <= 0).then(None).otherwise(cov / denom)


def ofi_correlation(prefix_a: str, prefix_b: str, window: int) -> pl.Expr:
    """Rolling corr(top-book OFI_A, OFI_B) with OFI = bid_sz_L1 − ask_sz_L1."""
    ofi_a = pl.col(f"{prefix_a}_bid_sz_L1") - pl.col(f"{prefix_a}_ask_sz_L1")
    ofi_b = pl.col(f"{prefix_b}_bid_sz_L1") - pl.col(f"{prefix_b}_ask_sz_L1")
    mean_a = ofi_a.rolling_mean(window_size=window)
    mean_b = ofi_b.rolling_mean(window_size=window)
    cov = ((ofi_a - mean_a) * (ofi_b - mean_b)).rolling_mean(window_size=window)
    var_a = ((ofi_a - mean_a) ** 2).rolling_mean(window_size=window)
    var_b = ((ofi_b - mean_b) ** 2).rolling_mean(window_size=window)
    denom = (var_a * var_b).sqrt()
    return pl.when(denom <= 0).then(None).otherwise(cov / denom)


# ---------------------------------------------------------------------------
# Pairwise state diffs
# ---------------------------------------------------------------------------

def microprice_diff(prefix_a: str, prefix_b: str) -> pl.Expr:
    """microprice_A − microprice_B. Needs L1 bid/ask price + size per symbol."""
    mp_a = (
        pl.col(f"{prefix_a}_bid_px_L1") * pl.col(f"{prefix_a}_ask_sz_L1")
        + pl.col(f"{prefix_a}_ask_px_L1") * pl.col(f"{prefix_a}_bid_sz_L1")
    ) / (pl.col(f"{prefix_a}_bid_sz_L1") + pl.col(f"{prefix_a}_ask_sz_L1") + EPS)
    mp_b = (
        pl.col(f"{prefix_b}_bid_px_L1") * pl.col(f"{prefix_b}_ask_sz_L1")
        + pl.col(f"{prefix_b}_ask_px_L1") * pl.col(f"{prefix_b}_bid_sz_L1")
    ) / (pl.col(f"{prefix_b}_bid_sz_L1") + pl.col(f"{prefix_b}_ask_sz_L1") + EPS)
    return mp_a - mp_b


def depth_imbalance_diff(prefix_a: str, prefix_b: str, depth: int = 5) -> pl.Expr:
    """Imbalance_A(depth) − Imbalance_B(depth).
    Each imbalance = (Σ bid_sz − Σ ask_sz) / (Σ bid_sz + Σ ask_sz) over 1..depth.
    """
    def _imb(pref: str) -> pl.Expr:
        b = [pl.col(f"{pref}_bid_sz_L{k}") for k in range(1, depth + 1)]
        a = [pl.col(f"{pref}_ask_sz_L{k}") for k in range(1, depth + 1)]
        b_sum = sum(b[1:], start=b[0])
        a_sum = sum(a[1:], start=a[0])
        return (b_sum - a_sum) / (b_sum + a_sum + EPS)
    return _imb(prefix_a) - _imb(prefix_b)


# ---------------------------------------------------------------------------
# Pairs / spread
# ---------------------------------------------------------------------------

def pairs_spread_zscore(prefix_a: str, prefix_b: str, window: int) -> pl.Expr:
    """Rolling z-score of (mid_A − mid_B). Classic pairs-trading signal."""
    spread = _mid(prefix_a) - _mid(prefix_b)
    return (spread - spread.rolling_mean(window_size=window)) / (
        spread.rolling_std(window_size=window) + EPS
    )


def relative_quoted_spread_diff(prefix_a: str, prefix_b: str) -> pl.Expr:
    """rs_A − rs_B where rs = (ask − bid) / mid (relative quoted spread)."""
    def _rs(p: str) -> pl.Expr:
        ask = pl.col(f"{p}_ask_px_L1")
        bid = pl.col(f"{p}_bid_px_L1")
        mid = (ask + bid) / 2
        return (ask - bid) / (mid + EPS)
    return _rs(prefix_a) - _rs(prefix_b)


# ---------------------------------------------------------------------------
# Volatility + lead-lag
# ---------------------------------------------------------------------------

def realized_volatility_ratio(
    prefix_a: str, prefix_b: str, window: int
) -> pl.Expr:
    """vol_A / vol_B (rolling std of mid log-returns, same window)."""
    ra = _log_return(prefix_a)
    rb = _log_return(prefix_b)
    vol_a = ra.rolling_std(window_size=window)
    vol_b = rb.rolling_std(window_size=window)
    return vol_a / (vol_b + EPS)


def price_lead_lag(prefix_a: str, prefix_b: str, lag: int = 1) -> pl.Expr:
    """Cross-product: ret_A · ret_B.shift(lag).

    Positive product = both moved together after lag bars (A leads B).
    This is the raw ingredient; caller can rolling-sum/mean for a smoothed version.
    """
    ra = _log_return(prefix_a)
    rb_lag = _log_return(prefix_b).shift(lag)
    return ra * rb_lag
