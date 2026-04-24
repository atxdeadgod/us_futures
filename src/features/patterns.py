"""Tier-7 pattern features — accumulation / fakeout / reversal detection.

Stateful rolling-window computations that operate on our 5-sec (or aggregated
15-min) bar schema. Framework-free: each function is a pl.Expr returning the
feature column.

Coverage (FEATURES.md T7.01-T7.12):
    T7.01 absorption_score
    T7.02 queue_replenishment_rate_l1         (NEEDS PRE-COMPUTED L1 DEPTH DIFFS)
    T7.03 volume_at_price_concentration
    T7.04 breakout_magnitude_up / _down
    T7.05 breakout_reversal_flag
    T7.06 post_breakout_flow_reversal          (stateful; simplified version)
    T7.07 spike_and_fade_volume
    T7.08 imbalance_persistence_runlength
    T7.09 cvd_price_divergence_flag           (uses §8.D RTH-bounded windows)
    T7.10 range_compression_ratio
    T7.11 round_number_pin_distance → already in engines.py
    T7.12 hidden_liquidity_rolling_ratio
"""
from __future__ import annotations

import polars as pl

EPS = 1e-9


# ---------------------------------------------------------------------------
# T7.01 Absorption score
# ---------------------------------------------------------------------------

def absorption_score(
    aggressor_dollar_col: pl.Expr,
    return_vol_col: pl.Expr,
    notional_col: pl.Expr,
    window: int = 20,
) -> pl.Expr:
    """T7.01: Σ|aggressor_dollar| / (return_vol × notional), rolling over `window`.

    High when sustained one-sided aggressor flow produces little price movement.
    Precursor to direction — someone's absorbing at a level.
    """
    abs_flow = aggressor_dollar_col.abs()
    num = abs_flow.rolling_sum(window_size=window)
    denom = (return_vol_col * notional_col + EPS).rolling_mean(window_size=window)
    return num / (denom * window + EPS)


# ---------------------------------------------------------------------------
# T7.03 Volume-at-price concentration
# ---------------------------------------------------------------------------

def volume_at_price_concentration(
    bar_volume_col: pl.Expr,
    at_price_volume_col: pl.Expr,
) -> pl.Expr:
    """T7.03: fraction of bar volume traded at (or very near) the bar's close price.

    Requires the caller to pre-compute `at_price_volume_col` at bar-build time
    (sum of trades within ±k ticks of close). If that's not available, pass
    large_trade_volume or another proxy. Range [0, 1].
    """
    return at_price_volume_col / (bar_volume_col + EPS)


# ---------------------------------------------------------------------------
# T7.04 Breakout magnitude
# ---------------------------------------------------------------------------

def breakout_magnitude_up(
    high_col: pl.Expr, atr_col: pl.Expr, lookback_bars: int = 30
) -> pl.Expr:
    """T7.04a: (current_high − max_high_over_prior_N_bars) / ATR.

    Positive = price made a new local high. Zero = didn't exceed prior high.
    Caller supplies ATR (e.g., from range_vol_parkinson × close, or a True-Range
    rolling mean).
    """
    prior_high = high_col.shift(1).rolling_max(window_size=lookback_bars)
    return pl.when(high_col > prior_high).then((high_col - prior_high) / (atr_col + EPS)).otherwise(0.0)


def breakout_magnitude_down(
    low_col: pl.Expr, atr_col: pl.Expr, lookback_bars: int = 30
) -> pl.Expr:
    """T7.04b: (min_low_over_prior_N − current_low) / ATR. Positive magnitude for new low."""
    prior_low = low_col.shift(1).rolling_min(window_size=lookback_bars)
    return pl.when(low_col < prior_low).then((prior_low - low_col) / (atr_col + EPS)).otherwise(0.0)


# ---------------------------------------------------------------------------
# T7.05 Breakout-reversal flag (Wyckoff upthrust / spring)
# ---------------------------------------------------------------------------

def breakout_reversal_up(
    high_col: pl.Expr, close_col: pl.Expr, atr_col: pl.Expr,
    lookback_bars: int = 30, reversal_atr: float = 0.5,
) -> pl.Expr:
    """T7.05: Wyckoff-upthrust flag.

    Bar breached prior N-bar high (high > prior_high) AND closed back below
    prior_high by more than `reversal_atr × ATR`. Classic bull-trap / fakeout.
    Returns Int8 (1/0).
    """
    prior_high = high_col.shift(1).rolling_max(window_size=lookback_bars)
    breached = high_col > prior_high
    reversed_back = close_col < prior_high - reversal_atr * atr_col
    return (breached & reversed_back).cast(pl.Int8)


def breakout_reversal_down(
    low_col: pl.Expr, close_col: pl.Expr, atr_col: pl.Expr,
    lookback_bars: int = 30, reversal_atr: float = 0.5,
) -> pl.Expr:
    """T7.05 mirror: Wyckoff spring. Bar broke below prior low AND closed back above."""
    prior_low = low_col.shift(1).rolling_min(window_size=lookback_bars)
    breached = low_col < prior_low
    reversed_back = close_col > prior_low + reversal_atr * atr_col
    return (breached & reversed_back).cast(pl.Int8)


# ---------------------------------------------------------------------------
# T7.06 Post-breakout flow reversal (simplified)
# ---------------------------------------------------------------------------

def post_breakout_flow_reversal(
    breakout_flag_col: pl.Expr, aggressor_sign_col: pl.Expr, lookforward_bars: int = 3
) -> pl.Expr:
    """T7.06: (simplified) After a breakout flag fires, sum aggressor signs over
    the next `lookforward_bars`. Negative value after an UPSIDE breakout → flow
    reversed (buyers didn't follow through).

    Pairs with breakout_reversal_up for the "breakout then flow reversal" composite.

    Caller passes: an already-computed breakout flag (e.g. breakout_magnitude_up > 0 cast int),
    and a per-bar aggregated aggressor signal (e.g., (buys_qty - sells_qty) / (buys+sells)).
    """
    # Forward-looking sum (shift by -lookforward_bars brings future bars back)
    forward_sum = aggressor_sign_col.rolling_sum(window_size=lookforward_bars).shift(-lookforward_bars)
    # Only emit when breakout flag is set; else 0
    return pl.when(breakout_flag_col > 0).then(forward_sum).otherwise(0.0)


# ---------------------------------------------------------------------------
# T7.07 Spike-and-fade volume
# ---------------------------------------------------------------------------

def spike_and_fade_volume(
    volume_col: pl.Expr, lookback_bars: int = 20, spike_multiplier: float = 3.0
) -> pl.Expr:
    """T7.07: 1 when current bar's volume > spike_multiplier × rolling_mean AND
    NEXT bar's volume < rolling_mean / spike_multiplier.

    Returns Int8 flag. Uses shift(-1) to look forward (causality ok for emitting
    a SIGNAL at time t+1; if used as a t-label, caller should be aware).
    """
    baseline = volume_col.rolling_mean(window_size=lookback_bars).shift(1)  # don't include current
    is_spike = volume_col > spike_multiplier * baseline
    next_vol = volume_col.shift(-1)
    is_fade = next_vol < baseline / spike_multiplier
    return (is_spike & is_fade).cast(pl.Int8)


# ---------------------------------------------------------------------------
# T7.08 Imbalance persistence run-length
# ---------------------------------------------------------------------------

def imbalance_persistence_runlength(
    imbalance_col: pl.Expr, window: int = 30
) -> pl.Expr:
    """T7.08: rolling sum of sign(imbalance) over `window`. Range [-window, window].

    Large positive = sustained buy-side dominance across N consecutive bars.
    Approximates "run length" without true run-detection overhead.
    """
    sign_col = (
        pl.when(imbalance_col > 0).then(1)
        .when(imbalance_col < 0).then(-1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    return sign_col.rolling_sum(window_size=window)


# ---------------------------------------------------------------------------
# T7.09 CVD / price divergence flag (RTH-bounded)
# ---------------------------------------------------------------------------

def cvd_price_divergence_up(
    cvd_rth_col: pl.Expr,
    price_high_col: pl.Expr,
    window: int = 30,
) -> pl.Expr:
    """T7.09: price made new rolling-high BUT CVD_rth did NOT make new rolling-high.

    Classic mid-freq mean-reversion signal: price drifting up on weakening flow
    = limit-sellers absorbing aggressive buys.

    NOTE: for true RTH-bounded window semantics (§8.D), caller should use the
    `rolling_rth_bounded` helper from engines.py and pass its output, then just
    compare via this function. This expression uses plain rolling_max.
    """
    price_high_roll = price_high_col.rolling_max(window_size=window).shift(1)
    cvd_high_roll = cvd_rth_col.rolling_max(window_size=window).shift(1)
    return ((price_high_col > price_high_roll) & (cvd_rth_col <= cvd_high_roll)).cast(pl.Int8)


def cvd_price_divergence_down(
    cvd_rth_col: pl.Expr,
    price_low_col: pl.Expr,
    window: int = 30,
) -> pl.Expr:
    """T7.09 mirror: price made new rolling-low but CVD_rth did not → bottoming."""
    price_low_roll = price_low_col.rolling_min(window_size=window).shift(1)
    cvd_low_roll = cvd_rth_col.rolling_min(window_size=window).shift(1)
    return ((price_low_col < price_low_roll) & (cvd_rth_col >= cvd_low_roll)).cast(pl.Int8)


# ---------------------------------------------------------------------------
# T7.10 Range compression
# ---------------------------------------------------------------------------

def range_compression_ratio(
    high_col: pl.Expr, low_col: pl.Expr, atr_col: pl.Expr,
    window: int = 20,
) -> pl.Expr:
    """T7.10: rolling_std(high − low) / ATR. Low values = tight range, pre-accumulation.

    Ratio < 0.5 typically means the last N bars had HLs well-clustered relative
    to the volatility baseline.
    """
    hl_range = high_col - low_col
    return hl_range.rolling_std(window_size=window) / (atr_col + EPS)


# ---------------------------------------------------------------------------
# T7.12 Hidden liquidity rolling ratio
# ---------------------------------------------------------------------------

def hidden_liquidity_rolling_ratio(
    hidden_absorption_col: pl.Expr, volume_col: pl.Expr, window: int = 30
) -> pl.Expr:
    """T7.12: rolling sum of hidden absorption volume / rolling sum of bar volume.

    Pairs with bars_exec.hidden_absorption_bars (T1.47) which emits the per-bar
    hidden_absorption_volume. This produces the rolling proportion over `window`.
    """
    return hidden_absorption_col.rolling_sum(window_size=window) / (
        volume_col.rolling_sum(window_size=window) + EPS
    )
