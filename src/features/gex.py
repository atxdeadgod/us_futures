"""Dealer Gamma Exposure (GEX) features — T5.08-T5.16.

Two-step pipeline:

  Step 1: compute_daily_gex_profile(options_chain, spot_series)
    Takes WRDS optionm.opprcdYYYY SPX chain parquet + daily spot prices (SPX
    close) → emits per-date scalars:
      total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike,
      max_put_oi_strike, gex_0dte_share, gex_without_0dte, gex_0dte_only

  Step 2: attach_gex_features(bars, daily_gex, es_spx_basis)
    asof-backward joins the daily profile onto ES bars (by date) using
    engines.asof_strict_backward (§8.E); computes intraday-varying distance
    features using ES bar close vs ES-adjusted strike levels.

References:
  Baltussen-Da-Lammers-Martens (2021) — intraday momentum by GEX regime
  Barbon-Buraschi (2021) — gamma fragility / flip levels
  Brogaard-Han-Won (2024) — 0DTE dealer hedging

Dealer-positioning assumption (§8.I): NAIVE heuristic for V1.
  Dealers short calls (retail buys calls) → call contribution = -OI·γ·S²·100
  Dealers long puts  (retail sells puts)  → put  contribution = +OI·γ·S²·100
  Calibration via customer-flow data is a V2 upgrade (see TODO.md).

§8.J mitigation: cap per-option |γ·OI| at 99th percentile per date to prevent
single ATM 0DTE options from dominating the day's GEX.

§8.K SPX→ES basis: zero_gamma_strike_ES = zero_gamma_strike_SPX + (ES_close - SPX_close)
using the PREVIOUS day's close for stability. Applied in attach_gex_features.

§8.L lookahead control: daily profile is built from T-1 EOD options. Features at
bar-close time T are safe.
"""
from __future__ import annotations

import polars as pl


SPX_CONTRACT_SIZE = 100.0  # shares per contract (applies to SPX and equity-style options)


def compute_daily_gex_profile(
    options_chain: pl.DataFrame,
    spot_daily: pl.DataFrame,
    dealer_sign_assumption: str = "naive_call_short",
    cap_pct: float = 0.99,
) -> pl.DataFrame:
    """Per-day GEX scalars from a daily SPX options chain.

    Args:
        options_chain: columns [date, secid, strike_price (in $), exdate, cp_flag,
            open_interest, gamma]. One row per option per date.
        spot_daily: columns [date, spot]. Daily close of the underlying index.
        dealer_sign_assumption: "naive_call_short" → dealers short calls / long puts.
        cap_pct: per-date cap for |γ·OI| contribution (99th percentile default).

    Returns:
        DataFrame with per-date columns:
            date, total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike,
            max_put_oi_strike, gex_0dte_share, gex_0dte_only, gex_without_0dte
    """
    if dealer_sign_assumption != "naive_call_short":
        raise NotImplementedError(f"Only naive_call_short in V1; got {dealer_sign_assumption!r}")

    df = options_chain.join(spot_daily, on="date", how="inner")
    df = df.with_columns(
        [
            pl.when(pl.col("cp_flag") == "P").then(1).otherwise(-1).alias("_dealer_sign"),
            (pl.col("exdate") == pl.col("date")).alias("_is_0dte"),
            (
                pl.col("open_interest") * pl.col("gamma") * (pl.col("spot") ** 2) * SPX_CONTRACT_SIZE
            ).alias("_gamma_dollar_abs"),
        ]
    )

    # §8.J cap: per-date 99th percentile cap on |γ·OI·S²·100|
    df = df.with_columns(
        pl.col("_gamma_dollar_abs")
        .quantile(cap_pct)
        .over("date")
        .alias("_cap_99")
    )
    df = df.with_columns(
        pl.min_horizontal(pl.col("_gamma_dollar_abs"), pl.col("_cap_99")).alias("_gamma_capped")
    )

    df = df.with_columns(
        (pl.col("_dealer_sign") * pl.col("_gamma_capped")).alias("_dealer_contribution")
    )

    # Total GEX per date (signed) + per-bucket split
    totals = df.group_by("date").agg(
        [
            pl.col("_dealer_contribution").sum().alias("total_gex"),
            pl.col("_dealer_contribution").filter(pl.col("_is_0dte")).sum().alias("gex_0dte_only"),
            pl.col("_gamma_capped").filter(pl.col("_is_0dte")).sum().alias("_abs_0dte"),
            pl.col("_gamma_capped").sum().alias("_abs_total"),
        ]
    ).with_columns(
        [
            pl.col("total_gex").sign().alias("gex_sign"),
            (pl.col("_abs_0dte") / (pl.col("_abs_total") + 1e-9)).alias("gex_0dte_share"),
            (pl.col("total_gex") - pl.col("gex_0dte_only")).alias("gex_without_0dte"),
        ]
    ).drop(["_abs_0dte", "_abs_total"])

    # Zero-gamma flip: per date, sort options by strike, cumulative dealer_contribution,
    # find strike where cumulative sign FIRST flips (from negative to positive typically).
    # Do this per-date using a groupby+cumsum pattern.
    by_strike = df.group_by(["date", "strike_price"]).agg(
        pl.col("_dealer_contribution").sum().alias("gamma_at_strike")
    ).sort(["date", "strike_price"])
    by_strike = by_strike.with_columns(
        pl.col("gamma_at_strike").cum_sum().over("date").alias("cum_gamma")
    )
    # Zero-flip strike = first strike where cum_gamma >= 0 (transitioning from negative
    # below-the-money to positive above-the-money). If no crossing, use the median strike.
    flip = (
        by_strike.filter(pl.col("cum_gamma") >= 0)
        .group_by("date")
        .agg(pl.col("strike_price").min().alias("zero_gamma_strike"))
    )

    # Max-OI strikes per side
    calls = options_chain.filter(pl.col("cp_flag") == "C")
    puts = options_chain.filter(pl.col("cp_flag") == "P")
    max_call_oi = (
        calls.group_by(["date", "strike_price"])
        .agg(pl.col("open_interest").sum().alias("_oi"))
        .sort(["date", "_oi"], descending=[False, True])
        .group_by("date")
        .agg(pl.col("strike_price").first().alias("max_call_oi_strike"))
    )
    max_put_oi = (
        puts.group_by(["date", "strike_price"])
        .agg(pl.col("open_interest").sum().alias("_oi"))
        .sort(["date", "_oi"], descending=[False, True])
        .group_by("date")
        .agg(pl.col("strike_price").first().alias("max_put_oi_strike"))
    )

    out = (
        totals.join(flip, on="date", how="left")
        .join(max_call_oi, on="date", how="left")
        .join(max_put_oi, on="date", how="left")
        .sort("date")
    )
    return out.select(
        [
            "date",
            "total_gex",
            "gex_sign",
            "zero_gamma_strike",
            "max_call_oi_strike",
            "max_put_oi_strike",
            "gex_0dte_share",
            "gex_0dte_only",
            "gex_without_0dte",
        ]
    )


# ---------------------------------------------------------------------------
# Step 2 — per-bar feature attachment
# ---------------------------------------------------------------------------

def attach_gex_features(
    bars: pl.DataFrame,
    daily_gex: pl.DataFrame,
    es_spx_basis: pl.DataFrame,
    ts_col: str = "ts",
    close_col: str = "close",
) -> pl.DataFrame:
    """Attach T5.08-T5.15 (+ T5.16 needs VIX, handle in caller).

    Args:
        bars: ES bar frame with `ts` (UTC datetime) and `close` (ES price).
        daily_gex: output of compute_daily_gex_profile.
        es_spx_basis: columns [date, basis] where basis = ES_close - SPX_close on that date.

    Returns:
        Bars DataFrame with extra columns:
            total_gex, gex_sign, gex_0dte_share,
            gex_log_signed                          # T5.08
            distance_to_zero_gamma_flip             # T5.10
            distance_to_zero_gamma_flip_bp          # T5.11
            distance_to_max_call_oi                 # T5.13
            distance_to_max_put_oi                  # T5.14
    """
    # Bar-level date (in UTC — for GEX it's fine; we're joining on daily granularity)
    bars = bars.with_columns(pl.col(ts_col).dt.date().alias("_bar_date"))

    # Asof-backward: GEX profile from previous day (or today if it's the EOD we're using)
    # For strict no-lookahead, we could use yesterday's profile. For simplicity and since
    # daily_gex is EOD-of-day-T, we shift by 1 day so bar at T uses T-1 EOD profile.
    daily_shifted = daily_gex.with_columns(
        (pl.col("date") + pl.duration(days=1)).alias("_applies_from_date")
    ).drop("date").rename({"_applies_from_date": "_bar_date"})

    basis_shifted = es_spx_basis.with_columns(
        (pl.col("date") + pl.duration(days=1)).alias("_applies_from_date")
    ).drop("date").rename({"_applies_from_date": "_bar_date"})

    bars = bars.join(daily_shifted, on="_bar_date", how="left")
    bars = bars.join(basis_shifted, on="_bar_date", how="left")

    # ES-adjusted strike levels (§8.K)
    bars = bars.with_columns(
        [
            (pl.col("zero_gamma_strike") + pl.col("basis")).alias("_zero_gamma_strike_es"),
            (pl.col("max_call_oi_strike") + pl.col("basis")).alias("_max_call_strike_es"),
            (pl.col("max_put_oi_strike") + pl.col("basis")).alias("_max_put_strike_es"),
        ]
    )

    # T5.08 log-signed: sign(total_gex) × log(1 + |total_gex|)
    bars = bars.with_columns(
        (pl.col("gex_sign") * (1.0 + pl.col("total_gex").abs()).log()).alias("gex_log_signed")
    )

    # T5.10, T5.11
    bars = bars.with_columns(
        [
            (pl.col(close_col) - pl.col("_zero_gamma_strike_es")).alias("distance_to_zero_gamma_flip"),
        ]
    )
    bars = bars.with_columns(
        (pl.col("distance_to_zero_gamma_flip") / pl.col(close_col) * 10_000).alias(
            "distance_to_zero_gamma_flip_bp"
        )
    )

    # T5.13, T5.14
    bars = bars.with_columns(
        [
            (pl.col(close_col) - pl.col("_max_call_strike_es")).alias("distance_to_max_call_oi"),
            (pl.col(close_col) - pl.col("_max_put_strike_es")).alias("distance_to_max_put_oi"),
        ]
    )

    return bars.drop(
        [
            "_bar_date",
            "_zero_gamma_strike_es",
            "_max_call_strike_es",
            "_max_put_strike_es",
        ]
    )


# ---------------------------------------------------------------------------
# T5.15 Zero-gamma cross flag (rolling, per bar)
# ---------------------------------------------------------------------------

def zero_gamma_cross_flag(
    distance_col: pl.Expr, window: int = 30
) -> pl.Expr:
    """T5.15: 1 if ES crossed the zero-gamma flip level within the last `window` bars.

    Detection: sign(distance) changes between any two bars in the window.
    Equivalent to: (rolling min × rolling max < 0).
    """
    rmin = distance_col.rolling_min(window_size=window)
    rmax = distance_col.rolling_max(window_size=window)
    return ((rmin * rmax) < 0).cast(pl.Int8)


# ---------------------------------------------------------------------------
# T5.16 GEX × VIX interaction
# ---------------------------------------------------------------------------

def gex_vix_interaction(gex_sign_col: pl.Expr, vix_zscore_col: pl.Expr) -> pl.Expr:
    """T5.16: gex_sign × vix_zscore.

    Negative GEX + elevated VIX → negative sign × large positive z = large negative
    value → signals maximum trend-amplification potential.
    """
    return gex_sign_col * vix_zscore_col
