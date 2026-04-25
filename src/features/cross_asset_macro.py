"""Cross-asset macro feature primitives + composites.

Two categories of utilities:

1. **Distributional normalization primitives** — apply to ANY base feature
   (returns, OFI, CVD, spread, vol surprise, etc.) computed by other modules:
     attach_mad_zscore     — outlier-robust time-series z-score (hour-partitioned)
     attach_gauss_rank_cs  — cross-sectional Gauss-Rank across multiple contracts

   These complement `tc_features.attach_tc_zscore` (mean/std-based time-series z).

2. **Cross-asset composites** that combine multiple instruments into a single
   feature, suitable for predicting equity-index direction:
     attach_synthetic_dxy_logret   — weighted FX log-returns (DXY-like)
     attach_rates_curve_spreads    — 2s5s, 5s10s, 2s10s, 10s30s, butterfly
     attach_rolling_correlation    — cross-asset rolling pearson corr (regime indicator)
     attach_risk_on_off_composite  — continuous score (gold↑ + bonds↑ + DXY↑ - equities↑)

The panel.py orchestrator calls existing microstructure modules to compute base
values, then applies these primitives + composites on the wide cross-asset frame.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

EPS = 1e-9


# ---------------------------------------------------------------------------
# 1A. Time-series MAD z-score (outlier-robust dual of TC z-score)
# ---------------------------------------------------------------------------

def attach_mad_zscore(
    bars: pl.DataFrame,
    value_col: str,
    lookback_days: int = 30,
    bar_minutes: int = 15,
    partition_minutes: int | None = None,
    ts_col: str = "ts",
    out_col: str | None = None,
) -> pl.DataFrame:
    """Attach `(value − rolling_median) / (1.4826 × rolling_MAD)` for `value_col`.

    Same partitioning pattern as `tc_features.attach_tc_zscore` (15-min bar-of-day,
    hour-conditional in US/Eastern), but uses median + MAD instead of mean + std.
    The 1.4826 factor makes MAD equal to std for normal-distributed data; for
    fat-tailed data the MAD z-score keeps outlier moves marked at full magnitude
    (whereas std-based z-score gets compressed when outliers inflate std).

    Args:
        value_col: source column to normalize
        lookback_days: window of same-bar-of-day samples to estimate location/scale
        bar_minutes: bar duration (15 for 15-min bars)
        partition_minutes: bucket granularity (default = bar_minutes)
        out_col: output column name (default `{value_col}_tc_madz`)
    """
    if partition_minutes is None:
        partition_minutes = bar_minutes
    if 60 % partition_minutes != 0:
        raise ValueError(f"partition_minutes={partition_minutes} must divide 60")
    if out_col is None:
        out_col = f"{value_col}_tc_madz"

    parts_per_hour = 60 // partition_minutes
    et = pl.col(ts_col).dt.convert_time_zone("US/Eastern")
    df = bars.with_columns(
        (et.dt.hour() * parts_per_hour + et.dt.minute() // partition_minutes).alias("_bod_madz")
    )
    df = df.with_columns(
        pl.col(value_col).rolling_quantile(
            quantile=0.5, window_size=lookback_days, interpolation="midpoint",
        ).over("_bod_madz").alias("_rmed")
    )
    df = df.with_columns(
        (pl.col(value_col) - pl.col("_rmed")).abs().alias("_abs_dev")
    )
    # Second rolling_quantile sees the first lookback_days-1 rows as null (rmed warmup).
    # Set min_periods=lookback_days//2 so MAD is available shortly after rmed is,
    # rather than requiring an additional full window of non-null abs_dev values.
    df = df.with_columns(
        pl.col("_abs_dev").rolling_quantile(
            quantile=0.5, window_size=lookback_days, interpolation="midpoint",
            min_samples=max(2, lookback_days // 2),
        ).over("_bod_madz").alias("_mad")
    )
    df = df.with_columns(
        ((pl.col(value_col) - pl.col("_rmed")) / (1.4826 * pl.col("_mad") + EPS)).alias(out_col)
    )
    return df.drop(["_bod_madz", "_rmed", "_abs_dev", "_mad"])


# ---------------------------------------------------------------------------
# 1B. Cross-sectional Gauss-Rank
# ---------------------------------------------------------------------------

def attach_gauss_rank_cs(
    panel: pl.DataFrame,
    value_cols: list[str],
    out_prefix: str = "gauss_rank_",
) -> pl.DataFrame:
    """Cross-sectional Gauss-Rank across `value_cols` at each row.

    For each row (i.e., each timestamp):
      1. Rank the values across `value_cols` (NaN-aware, ignored in ranking)
      2. Convert rank to quantile: (rank + 1) / (n_valid + 1)  ∈ (0, 1)
      3. Apply inverse normal CDF → standard normal-distributed values

    Output: one column per input (`{out_prefix}{value_col}`) holding the
    Gauss-Rank for that contract at that ts. Bounded to roughly ±3 sigma,
    symmetric around 0, robust to outliers — ideal for tree-model features.

    Use case: with `value_cols=[log_return_ES, log_return_NQ, ..., log_return_GC, ...]`
    (one per contract at the same ts), the output marks "where in the cross-section
    is THIS contract's return at this moment" on a normal-shaped scale.
    """
    if not value_cols:
        return panel
    arr = panel.select(value_cols).to_numpy().astype(np.float64)
    n_rows, n_cols = arr.shape
    out = np.full_like(arr, np.nan)
    for i in range(n_rows):
        row = arr[i]
        valid = ~np.isnan(row)
        n = int(valid.sum())
        if n < 2:
            continue
        # rank within valid values: 0..n-1, ties broken by argsort order
        ranks = row[valid].argsort().argsort()
        quantiles = (ranks + 1) / (n + 1)
        out[i, valid] = norm.ppf(quantiles)
    new_cols = [
        pl.Series(f"{out_prefix}{c}", out[:, i], dtype=pl.Float64)
        for i, c in enumerate(value_cols)
    ]
    return panel.with_columns(new_cols)


# ---------------------------------------------------------------------------
# 2A. Synthetic DXY (weighted FX log-returns from CME currency futures)
# ---------------------------------------------------------------------------

# Original DXY composite weights (sum = 1.0 across 6 currencies).
# CME futures cover 92.2% of weight (EUR, JPY, GBP, CAD); SEK and CHF excluded.
_DXY_RAW_WEIGHTS = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091}
_DXY_TOTAL_RAW = sum(_DXY_RAW_WEIGHTS.values())
DXY_WEIGHTS = {k: v / _DXY_TOTAL_RAW for k, v in _DXY_RAW_WEIGHTS.items()}


def attach_synthetic_dxy_logret(
    panel: pl.DataFrame,
    eur_logret_col: str,
    jpy_logret_col: str,
    gbp_logret_col: str,
    cad_logret_col: str,
    out_col: str = "synthetic_dxy_logret",
) -> pl.DataFrame:
    """Synthetic DXY log-return ≈ weighted negative sum of CME currency futures returns.

    CME currency futures (6E, 6J, 6B, 6C) all quote with USD as the numerator
    (e.g., USD per EUR for 6E). DXY uses the USD-denominator basket convention.
    A USD STRENGTH event = each foreign currency falls vs USD = each CME future
    falls. So synthetic DXY log-return ≈ NEGATIVE weighted sum of futures returns.

    Renormalized weights for the 4 available currencies:
      EUR = 0.625, JPY = 0.148, GBP = 0.129, CAD = 0.099  (sum = 1.0)
    """
    return panel.with_columns(
        (
            -DXY_WEIGHTS["EUR"] * pl.col(eur_logret_col)
            - DXY_WEIGHTS["JPY"] * pl.col(jpy_logret_col)
            - DXY_WEIGHTS["GBP"] * pl.col(gbp_logret_col)
            - DXY_WEIGHTS["CAD"] * pl.col(cad_logret_col)
        ).alias(out_col)
    )


# ---------------------------------------------------------------------------
# 2B. Rates curve spreads (slope, curvature) from Treasury futures returns
# ---------------------------------------------------------------------------

def attach_rates_curve_spreads(
    panel: pl.DataFrame,
    zt_logret_col: str,  # 2y futures return
    zf_logret_col: str,  # 5y
    zn_logret_col: str,  # 10y
    zb_logret_col: str,  # 30y
) -> pl.DataFrame:
    """Slope + curvature features from Treasury futures log-returns.

    Note: futures prices are INVERSE to yields. A slope-STEEPENING event
    (long-end yields rise more than short-end) shows up as long-end FUTURES
    falling more than short-end futures, i.e., (long_logret − short_logret) < 0.
    Whether that maps to "+ slope" or "− slope" in conventional yield-space depends
    on sign convention — the model can learn either; we just emit the raw spreads.

    Output columns:
        slope_2s5s_logret    = log_ret(ZF) − log_ret(ZT)
        slope_5s10s_logret   = log_ret(ZN) − log_ret(ZF)
        slope_2s10s_logret   = log_ret(ZN) − log_ret(ZT)
        slope_10s30s_logret  = log_ret(ZB) − log_ret(ZN)
        butterfly_2s5s10s    = log_ret(ZN) − 2·log_ret(ZF) + log_ret(ZT)  (curvature)
    """
    return panel.with_columns([
        (pl.col(zf_logret_col) - pl.col(zt_logret_col)).alias("slope_2s5s_logret"),
        (pl.col(zn_logret_col) - pl.col(zf_logret_col)).alias("slope_5s10s_logret"),
        (pl.col(zn_logret_col) - pl.col(zt_logret_col)).alias("slope_2s10s_logret"),
        (pl.col(zb_logret_col) - pl.col(zn_logret_col)).alias("slope_10s30s_logret"),
        (
            pl.col(zn_logret_col)
            - 2 * pl.col(zf_logret_col)
            + pl.col(zt_logret_col)
        ).alias("butterfly_2s5s10s"),
    ])


# ---------------------------------------------------------------------------
# 2C. Cross-asset rolling correlation (regime indicator)
# ---------------------------------------------------------------------------

def attach_rolling_correlation(
    panel: pl.DataFrame,
    col_a: str,
    col_b: str,
    window: int = 60,
    out_col: str | None = None,
) -> pl.DataFrame:
    """Rolling pearson correlation of two return columns over `window` bars.

    Captures regime: high corr means assets are syncing (common-factor driven);
    low / negative corr means decorrelation (either divergence or risk-off
    flight from one to the other).
    """
    if out_col is None:
        out_col = f"corr_{col_a}_{col_b}_w{window}"
    return panel.with_columns(
        pl.rolling_corr(pl.col(col_a), pl.col(col_b), window_size=window).alias(out_col)
    )


# ---------------------------------------------------------------------------
# 2D. Risk-on / off composite (continuous, weighted z-score sum)
# ---------------------------------------------------------------------------

def attach_risk_on_off_composite(
    panel: pl.DataFrame,
    gold_z_col: str,
    dxy_z_col: str,
    bond_z_cols: list[str],
    equity_z_cols: list[str],
    out_col: str = "risk_off_score",
) -> pl.DataFrame:
    """Continuous risk-off composite from already z-scored returns.

    Risk-OFF event: gold ↑ AND bonds ↑ AND DXY ↑ AND equities ↓.
    Composite score (high = risk-off, low = risk-on):

        score = z(gold) + mean(z(bonds)) + z(dxy) - mean(z(equities))

    Inputs MUST already be z-scored (TC z or MAD z) — the composite is a weighted
    sum, not a thresholded indicator. Tree models then find their own splits on
    the continuous score (and the underlying z-scores remain available too).
    """
    return panel.with_columns(
        (
            pl.col(gold_z_col)
            + pl.mean_horizontal(*[pl.col(c) for c in bond_z_cols])
            + pl.col(dxy_z_col)
            - pl.mean_horizontal(*[pl.col(c) for c in equity_z_cols])
        ).alias(out_col)
    )
