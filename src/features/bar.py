"""Bar-single features — pure polars expressions.

Extracted from feature-factory (bar/single/*). Framework-free.

Convention: each function takes column expressions (typically pl.col objects)
and a window/horizon parameter, returns a pl.Expr.

Features covered (20 of the T2 + T1.30 + T6 set):
  T2.01  log_return            (close, horizon=1)
  T2.02  cumulative_return     (close, anchor="session_open"|"running")
  T2.05  vwap_return           (high, low, close, volume, window)
  T2.06  return_autocorrelation(ret, lag, window)
  T2.07  realized_volatility   (ret, window, method="std"|"ewma")
  T2.10  range_vol_parkinson   (high, low, window)
  T2.10  range_vol_gk          (open, high, low, close, window)
  T2.18  jump_indicator        (ret, vol, threshold, output)
  T2.20  volatility_ratio      (short_vol_col, long_vol_col)
  T2.21  volume_surprise       (log_volume, window)
  T2.22  turnover              (volume, shares_outstanding, window)
  T2.23  price_volume_correlation (ret, log_vol, window)
  T2.24  price_impact_slope    (ret, log_vol, window)
  T1.30  amihud_illiquidity    (ret, volume, close, window)

  Helpers: log_return_horizon (parametric horizon),
           log_volume (log(volume+1))
"""
from __future__ import annotations

import math

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def log_return(price_col: pl.Expr, horizon: int = 1) -> pl.Expr:
    """T2.01: log(price / price.shift(horizon))."""
    return (price_col / price_col.shift(horizon)).log()


def cumulative_return(price_col: pl.Expr, anchor_col: pl.Expr | None = None) -> pl.Expr:
    """T2.02: log(price / anchor_price).
    If anchor_col is None, uses first-ever price (cum from start of series).
    """
    anchor = anchor_col if anchor_col is not None else price_col.first()
    return (price_col / anchor).log()


def log_volume(volume_col: pl.Expr) -> pl.Expr:
    """Helper: log(1 + volume), used by VolumeSurprise / PIS / PVC."""
    return (1 + volume_col).log()


# ---------------------------------------------------------------------------
# Realized volatility family
# ---------------------------------------------------------------------------

def realized_volatility(
    ret_col: pl.Expr, window: int, method: str = "std"
) -> pl.Expr:
    """T2.07/08/09: rolling realized vol.
        method='std'  → rolling_std(window)
        method='ewma' → sqrt of EWMA of squared returns (span=window)
    """
    if method == "std":
        return ret_col.rolling_std(window_size=window, ddof=0)
    if method == "ewma":
        return ret_col.pow(2).ewm_mean(span=window, adjust=False).sqrt()
    raise ValueError(f"method must be 'std' or 'ewma', got {method!r}")


def range_vol_parkinson(
    high_col: pl.Expr, low_col: pl.Expr, window: int
) -> pl.Expr:
    """T2.10a Parkinson: sqrt((1/(4·ln2)) · rolling_mean(ln²(H/L)))."""
    log_hl_sq = (high_col / low_col).log().pow(2)
    rolling = log_hl_sq.rolling_mean(window_size=window)
    base = rolling / (4 * math.log(2))
    return pl.when(base < 0).then(0.0).otherwise(base).sqrt()


def range_vol_gk(
    open_col: pl.Expr, high_col: pl.Expr, low_col: pl.Expr, close_col: pl.Expr, window: int
) -> pl.Expr:
    """T2.10b Garman-Klass: sqrt(rolling_mean(0.5·ln²(H/L) - (2ln2-1)·ln²(C/O)))."""
    log_hl_sq = (high_col / low_col).log().pow(2)
    log_co_sq = (close_col / open_col).log().pow(2)
    base = (0.5 * log_hl_sq - (2 * math.log(2) - 1) * log_co_sq).rolling_mean(window_size=window)
    return pl.when(base < 0).then(0.0).otherwise(base).sqrt()


def volatility_ratio(short_vol_col: pl.Expr, long_vol_col: pl.Expr) -> pl.Expr:
    """T2.20: ratio of short-window vol to long-window vol."""
    return short_vol_col / (long_vol_col + EPS)


# ---------------------------------------------------------------------------
# Jumps / autocorrelation
# ---------------------------------------------------------------------------

def jump_indicator(
    ret_col: pl.Expr, vol_col: pl.Expr, threshold: float = 3.0, output: str = "flag"
) -> pl.Expr:
    """T2.18: z-score of current return against PREVIOUS bar's vol.
        output='flag' → 1 if |zscore|>threshold else 0
        output='zscore' → raw z-score
    """
    zscore = ret_col / (vol_col.shift(1) + EPS)
    if output == "flag":
        return (zscore.abs() > threshold).cast(pl.Int8)
    if output == "zscore":
        return zscore
    raise ValueError(f"output must be flag|zscore, got {output!r}")


def return_autocorrelation(ret_col: pl.Expr, lag: int = 1, window: int = 20) -> pl.Expr:
    """T2.06: rolling Pearson correlation of returns with their lagged series."""
    x = ret_col
    y = ret_col.shift(lag)
    mean_x = x.rolling_mean(window_size=window)
    mean_y = y.rolling_mean(window_size=window)
    cov_xy = ((x - mean_x) * (y - mean_y)).rolling_mean(window_size=window)
    var_x = ((x - mean_x).pow(2)).rolling_mean(window_size=window)
    var_y = ((y - mean_y).pow(2)).rolling_mean(window_size=window)
    denom = (var_x * var_y).sqrt()
    return pl.when(denom.is_null() | (denom <= 0)).then(None).otherwise(cov_xy / denom)


# ---------------------------------------------------------------------------
# Volume / VWAP / illiquidity
# ---------------------------------------------------------------------------

def vwap_return(
    high_col: pl.Expr, low_col: pl.Expr, close_col: pl.Expr, volume_col: pl.Expr, window: int
) -> pl.Expr:
    """T2.05: log-return of rolling-window VWAP (typical_price = (H+L+C)/3)."""
    typical = (high_col + low_col + close_col) / 3
    tp_vol = typical * volume_col
    vol_sum = volume_col.rolling_sum(window_size=window)
    vwap = tp_vol.rolling_sum(window_size=window) / (vol_sum + EPS)
    ratio = vwap / vwap.shift(1)
    return pl.when(
        (vwap.is_null())
        | (vwap.shift(1).is_null())
        | (vol_sum <= EPS)
        | (vol_sum.shift(1) <= EPS)
        | (~ratio.is_finite())
        | (ratio <= 0)
    ).then(None).otherwise(ratio.log())


def volume_surprise(log_volume_col: pl.Expr, window: int) -> pl.Expr:
    """T2.21: rolling z-score of log-volume (shift by 1 to avoid leak)."""
    mean_ = log_volume_col.rolling_mean(window_size=window).shift(1)
    std_ = log_volume_col.rolling_std(window_size=window, ddof=0).shift(1)
    return (log_volume_col - mean_) / (std_ + EPS)


def turnover(volume_col: pl.Expr, shares_outstanding_col: pl.Expr, window: int) -> pl.Expr:
    """T2.22: rolling-sum(volume) / shares_outstanding. Futures proxy uses OI instead."""
    return volume_col.rolling_sum(window_size=window) / (shares_outstanding_col + EPS)


def amihud_illiquidity(
    ret_col: pl.Expr, volume_col: pl.Expr, close_col: pl.Expr, window: int
) -> pl.Expr:
    """T1.30: rolling mean of |ret| / (volume · close). Higher = more illiquid."""
    illiq = ret_col.abs() / (volume_col * close_col + EPS)
    return illiq.rolling_mean(window_size=window)


def price_volume_correlation(
    ret_col: pl.Expr, log_volume_col: pl.Expr, window: int
) -> pl.Expr:
    """T2.23: rolling Pearson(returns, log-volume)."""
    x = ret_col
    y = log_volume_col
    mean_x = x.rolling_mean(window_size=window)
    mean_y = y.rolling_mean(window_size=window)
    cov_xy = ((x - mean_x) * (y - mean_y)).rolling_mean(window_size=window)
    var_x = ((x - mean_x).pow(2)).rolling_mean(window_size=window)
    var_y = ((y - mean_y).pow(2)).rolling_mean(window_size=window)
    denom = (var_x * var_y).sqrt()
    return pl.when(denom <= 0).then(None).otherwise(cov_xy / denom)


def price_impact_slope(
    ret_col: pl.Expr, log_volume_col: pl.Expr, window: int
) -> pl.Expr:
    """T2.24: rolling OLS slope of returns regressed on log-volume.
    β = Cov(ret, log_vol) / Var(log_vol). Higher |β| = bigger price move per unit flow.
    """
    x = log_volume_col
    y = ret_col
    mean_x = x.rolling_mean(window_size=window)
    mean_y = y.rolling_mean(window_size=window)
    cov_xy = ((x - mean_x) * (y - mean_y)).rolling_mean(window_size=window)
    var_x = ((x - mean_x).pow(2)).rolling_mean(window_size=window)
    return pl.when(var_x <= 0).then(None).otherwise(cov_xy / var_x)


# ---------------------------------------------------------------------------
# Calendar dummies (T6.02, T6.03)
# ---------------------------------------------------------------------------

def is_monday(ts_col: pl.Expr) -> pl.Expr:
    """T6.02: Monday flag. Polars dt.weekday() uses ISO convention (Mon=1..Sun=7)."""
    return (ts_col.dt.weekday() == 1).cast(pl.Int8)


def is_friday(ts_col: pl.Expr) -> pl.Expr:
    """T6.02: Friday flag. Polars dt.weekday() uses ISO (Fri=5)."""
    return (ts_col.dt.weekday() == 5).cast(pl.Int8)


def is_month_start(ts_col: pl.Expr) -> pl.Expr:
    """T6.03: first day of month."""
    return (ts_col.dt.day() == 1).cast(pl.Int8)


def is_month_end(ts_col: pl.Expr) -> pl.Expr:
    """T6.03: last day of month.
    Polars has no direct 'month_end' — compute via next-day-month-change.
    """
    # last day of month = day when (ts + 1 day).month != ts.month
    return (
        (ts_col.dt.offset_by("1d").dt.month() != ts_col.dt.month()).cast(pl.Int8)
    )


# ---------------------------------------------------------------------------
# Temporal (T6.04, T6.07)
# ---------------------------------------------------------------------------

def minute_of_day(ts_col: pl.Expr) -> pl.Expr:
    """T6.04: integer [0..1439]. Int32-safe for hour*60 to avoid i8 overflow.
    ts_col is assumed to be in desired timezone already (convert beforehand).
    """
    return (
        ts_col.dt.hour().cast(pl.Int32) * 60
        + ts_col.dt.minute().cast(pl.Int32)
    )


def settlement_window_flag(ts_col_et: pl.Expr) -> pl.Expr:
    """T6.07: flag between 15:45 and 16:00 ET.
    `ts_col_et` must already be in America/New_York tz (use dt.convert_time_zone).
    """
    mod = (
        ts_col_et.dt.hour().cast(pl.Int32) * 60
        + ts_col_et.dt.minute().cast(pl.Int32)
    )
    return mod.is_between(945, 960).cast(pl.Int8)  # [15:45, 16:00) in minutes


# ---------------------------------------------------------------------------
# VWAP deviation (T2.29 NEW)
# ---------------------------------------------------------------------------

def vwap_deviation(close_col: pl.Expr, session_vwap_col: pl.Expr) -> pl.Expr:
    """T2.29: close − session_vwap.
    Positive = bar closed above the session volume-weighted average price.
    """
    return close_col - session_vwap_col
