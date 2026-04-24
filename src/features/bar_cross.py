"""Bar-level cross-asset / peer-group features — pure polars expressions.

Assumes callers have assembled a WIDE panel DataFrame where each symbol has its
own column (e.g. `ES_close`, `NQ_close`, `RTY_close`) at matching timestamps.
These functions either take explicit column-name lists (for peer-group metrics)
or single-column expressions (for pairwise lead-lag).

Covered (from FEATURES.md Tier 3 + cross-sectional helpers):
  T3.01-T3.07 bar_lead_lag_return     leader_col, lag
  T3.10 lead_lag_asymmetry_score       A→B vs B→A cross-correlation asymmetry
  return_dispersion                    std-across-peers, rolling-mean smoothed
  cross_sectional_return_zscore        (target − peer_mean) / peer_std
  breadth                              frac of peers positive
  leader_laggard_spread                max_return − min_return
  return_concentration                 HHI-like on returns across peers

Deferred (need OLS / state-space; separate module bar_cross_regression.py V2):
  factor_model_residual
  idiosyncratic_volatility
  cointegration_zscore
  ou_half_life
  residual_*  (band position, dispersion, shock repair)
"""
from __future__ import annotations

from typing import Iterable

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# T3.01-T3.07  Lead-lag return
# ---------------------------------------------------------------------------

def bar_lead_lag_return(leader_price_col: pl.Expr, lag: int = 1) -> pl.Expr:
    """T3.01-T3.07: leader's log-return, shifted by `lag` bars.

    Caller attaches to follower's frame as a predictor. Assumes leader and
    follower are ALIGNED by timestamp (same bar index).
    """
    assert lag >= 1, "lag must be >= 1"
    leader_return = (leader_price_col / leader_price_col.shift(1)).log()
    return leader_return.shift(lag)


# ---------------------------------------------------------------------------
# Peer-group dispersion and breadth
# ---------------------------------------------------------------------------

def _return_cols_from_prices(price_cols: list[str], horizon: int = 1) -> list[pl.Expr]:
    """Helper: list of log-return expressions for each price column."""
    return [
        (pl.col(c) / pl.col(c).shift(horizon)).log().alias(f"_ret_{c}")
        for c in price_cols
    ]


def return_dispersion(
    price_cols: list[str], window: int, return_horizon: int = 1
) -> pl.Expr:
    """Cross-sectional return-dispersion, rolling-mean smoothed.

    For each row: compute log-return for each symbol, then std across symbols,
    then rolling-mean over `window` bars.
    """
    rets = _return_cols_from_prices(price_cols, return_horizon)
    # std across peers per row via horizontal aggregation
    # Polars doesn't have pl.std_horizontal (yet), so compute as sqrt(var horizontal)
    # var = mean(x²) - mean(x)²
    means = [r for r in rets]
    mean_h = pl.mean_horizontal(means)
    # For sample std: need population std or sample — use population (ddof=0) for simplicity
    sq_devs = [(r - mean_h) ** 2 for r in rets]
    var_h = pl.mean_horizontal(sq_devs)
    std_h = var_h.sqrt()
    return std_h.rolling_mean(window_size=window)


def cross_sectional_return_zscore(
    target_price_col: str,
    peer_price_cols: list[str],
    return_horizon: int = 1,
) -> pl.Expr:
    """Cross-sectional z-score of target's return vs peer-group at each bar.

    z = (target_ret − peer_mean) / peer_std.
    """
    assert target_price_col in peer_price_cols or True, "target is optional in peer list"
    tgt_ret = (pl.col(target_price_col) / pl.col(target_price_col).shift(return_horizon)).log()
    peer_rets = _return_cols_from_prices(peer_price_cols, return_horizon)
    peer_mean = pl.mean_horizontal(peer_rets)
    sq = [(r - peer_mean) ** 2 for r in peer_rets]
    peer_std = pl.mean_horizontal(sq).sqrt()
    return (tgt_ret - peer_mean) / (peer_std + EPS)


def breadth(price_cols: list[str], return_horizon: int = 1) -> pl.Expr:
    """Fraction of peers with positive bar return (range [0, 1]).

    High = broad rally; low = broad sell-off.
    """
    rets = _return_cols_from_prices(price_cols, return_horizon)
    positive_flags = [(r > 0).cast(pl.Int64) for r in rets]
    return pl.sum_horizontal(positive_flags) / len(price_cols)


def leader_laggard_spread(price_cols: list[str], return_horizon: int = 1) -> pl.Expr:
    """Max peer return − min peer return at each bar.

    High = rotation/dispersion; low = everyone moving together.
    """
    rets = _return_cols_from_prices(price_cols, return_horizon)
    return pl.max_horizontal(rets) - pl.min_horizontal(rets)


def return_concentration(price_cols: list[str], return_horizon: int = 1) -> pl.Expr:
    """HHI-like concentration of absolute returns.

    out = Σ ret²_i / (Σ |ret_i|)². 1.0 = one dominant mover, 1/N = uniform.
    """
    rets = _return_cols_from_prices(price_cols, return_horizon)
    abs_rets = [r.abs() for r in rets]
    sq_sum = pl.sum_horizontal([r ** 2 for r in rets])
    abs_sum_sq = pl.sum_horizontal(abs_rets) ** 2
    return sq_sum / (abs_sum_sq + EPS)


# ---------------------------------------------------------------------------
# T3.10  Lead-Lag Asymmetry Score
# ---------------------------------------------------------------------------

def lead_lag_asymmetry(
    a_col: str, b_col: str, lag: int = 1, window: int = 60
) -> pl.Expr:
    """T3.10 simplified: corr(A_return, B_return.shift(lag)) − corr(A_return.shift(lag), B_return).

    Positive → A leads B; negative → B leads A.
    Uses rolling Pearson correlation.
    """
    a_ret = (pl.col(a_col) / pl.col(a_col).shift(1)).log()
    b_ret = (pl.col(b_col) / pl.col(b_col).shift(1)).log()

    def _roll_corr(x: pl.Expr, y: pl.Expr) -> pl.Expr:
        mx = x.rolling_mean(window_size=window)
        my = y.rolling_mean(window_size=window)
        cov = ((x - mx) * (y - my)).rolling_mean(window_size=window)
        vx = ((x - mx) ** 2).rolling_mean(window_size=window)
        vy = ((y - my) ** 2).rolling_mean(window_size=window)
        denom = (vx * vy).sqrt()
        return pl.when(denom <= 0).then(None).otherwise(cov / denom)

    ab = _roll_corr(a_ret, b_ret.shift(lag))
    ba = _roll_corr(a_ret.shift(lag), b_ret)
    return ab - ba
