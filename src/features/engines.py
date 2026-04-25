"""Reusable feature-computation engines with §8 implementation-trap fixes baked in.

Each engine is a pure function on numpy / polars. No I/O, no paths, no state outside
the call. Unit tests in `tests/test_engines.py` cover every §8 trap.

Engines:
    ffd_weights, fracdiff_series, fracdiff_auto_d        (§8.A)
    vpin_volume_buckets                                   (§8.B)
    hawkes_intensity_recursive                            (§8.C)
    cvd_with_dual_reset, rolling_rth_bounded              (§8.D)
    asof_strict_backward                                  (§8.E helper)
    round_number_pin_distance                             (§8.F)

Not here (call sites live elsewhere):
    - Algoseek session-warm flag (§8.G: applied during bar builder)
    - Cancel-proxy from MBP-10 snapshots (§8.M: in bars.py)
    - GEX profile (§8.I-L: in a dedicated gex.py, needs the SPX chain data)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# §8.A  Fixed-Width Window Fractional Differencing (AFML Ch. 5)
# ---------------------------------------------------------------------------

def ffd_weights(d: float, tau: float = 1e-5, max_k: int = 10_000) -> np.ndarray:
    """Binomial weights for fractional differencing, truncated at |w_k| < tau.

    Returns weights in order [w_0, w_1, ..., w_K] where
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Suitable as the b-coefficients of a causal FIR filter applied with lfilter.
    """
    if d <= 0:
        # d=0 is identity. Return [1.0].
        return np.asarray([1.0])
    w = [1.0]
    k = 1
    while k < max_k:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < tau:
            break
        w.append(w_next)
        k += 1
    return np.asarray(w, dtype=np.float64)


def fracdiff_series(x: pl.Series | np.ndarray, d: float, tau: float = 1e-5) -> pl.Series:
    """FFD applied as causal FIR. First (len(weights)-1) bars are NaN (startup transient).

    Note: uses scipy.signal.lfilter (causal), NOT convolve (non-causal default).
    """
    arr = x.to_numpy() if isinstance(x, pl.Series) else np.asarray(x, dtype=np.float64)
    arr = arr.astype(np.float64)
    weights = ffd_weights(d, tau)
    out = lfilter(weights, [1.0], arr)
    out[: len(weights) - 1] = np.nan
    return pl.Series(out)


def fracdiff_auto_d(
    x: pl.Series,
    p_value: float = 0.01,
    d_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    tau: float = 1e-5,
) -> tuple[pl.Series, float]:
    """Return (fracdiff series, chosen d). Picks smallest d in d_grid whose ADF test
    rejects unit-root at the given p-value. Falls back to d=1 if none pass.
    """
    from statsmodels.tsa.stattools import adfuller

    arr = x.to_numpy() if isinstance(x, pl.Series) else np.asarray(x)
    for d in d_grid:
        fd = fracdiff_series(pl.Series(arr), d, tau)
        valid = fd.drop_nulls().to_numpy()
        if len(valid) < 100:
            continue
        try:
            p = adfuller(valid, autolag="AIC")[1]
            if p < p_value:
                return fd, d
        except Exception:
            continue
    return fracdiff_series(pl.Series(arr), 1.0, tau), 1.0


# ---------------------------------------------------------------------------
# §8.B  VPIN volume buckets (Easley-Lopez de Prado, with Algoseek aggressor flag)
# ---------------------------------------------------------------------------

def vpin_volume_buckets(
    sub_bars: pl.DataFrame,
    bucket_size: int,
    ts_col: str = "ts",
    buys_col: str = "buys_qty",
    sells_col: str = "sells_qty",
    keep_partial: bool = False,
) -> pl.DataFrame:
    """Accumulate sub-bar signed volumes into volume buckets; emit VPIN per bucket.

    VPIN = |Σ buys − Σ sells| / Σ (buys + sells), over each bucket.

    Returns DataFrame with columns:
        bucket_close_ts, vpin, bucket_volume, n_sub_bars

    Partial bucket at end-of-input is discarded unless keep_partial=True.
    """
    ts_arr = sub_bars[ts_col].to_numpy()
    buys = sub_bars[buys_col].to_numpy().astype(np.int64)
    sells = sub_bars[sells_col].to_numpy().astype(np.int64)
    total_vol = buys + sells
    signed = buys - sells

    closes: list = []
    vpins: list = []
    vols: list = []
    counts: list = []

    cum_total = 0
    cum_signed = 0
    start_i = 0

    for i in range(len(sub_bars)):
        cum_total += total_vol[i]
        cum_signed += signed[i]
        if cum_total >= bucket_size:
            vpin = abs(cum_signed) / cum_total if cum_total > 0 else 0.0
            closes.append(ts_arr[i])
            vpins.append(float(vpin))
            vols.append(int(cum_total))
            counts.append(i - start_i + 1)
            start_i = i + 1
            cum_total = 0
            cum_signed = 0

    if keep_partial and cum_total > 0:
        vpin = abs(cum_signed) / cum_total
        closes.append(ts_arr[-1])
        vpins.append(float(vpin))
        vols.append(int(cum_total))
        counts.append(len(sub_bars) - start_i)

    # Build bucket_close_ts with the source's exact ts dtype. Without this,
    # polars infers `Datetime[μs]` from a list of numpy datetime64[ns] scalars
    # and silently misinterprets ns values as μs (year 56144 garbage). Going
    # through np.asarray preserves the datetime64 dtype, then we cast to the
    # source's exact dtype (carries the timezone, which numpy doesn't).
    ts_dtype = sub_bars.schema[ts_col]
    closes_arr = np.asarray(closes) if closes else np.array([], dtype="datetime64[ns]")
    return pl.DataFrame(
        {
            "bucket_close_ts": pl.Series("bucket_close_ts", closes_arr).cast(ts_dtype),
            "vpin": vpins,
            "bucket_volume": vols,
            "n_sub_bars": counts,
        }
    )


# ---------------------------------------------------------------------------
# §8.C  Hawkes intensity — recursive update with actual Δt
# ---------------------------------------------------------------------------

def hawkes_intensity_recursive(
    ts_seconds: np.ndarray,
    buys_volume: np.ndarray,
    sells_volume: np.ndarray,
    hl_seconds: float,
    session_reset_ts: np.ndarray | None = None,
    warmup_factor: float = 5.0,
) -> dict[str, np.ndarray]:
    """Recursive Hawkes self-exciting intensity for aggressor-signed volume.

    λ_t = λ_{t-1} · exp(-β · Δt_actual) + N_t
    β = ln(2) / hl_seconds

    Δt uses actual elapsed time between consecutive timestamps (not assumed 5s).
    `session_reset_ts` (optional): timestamps at which both intensities zero-reset
    (e.g., Algoseek session warmup start).

    Returns dict with per-row arrays: lambda_buy, lambda_sell, imbalance, is_warm.
    `is_warm` is True only after `warmup_factor × hl_seconds` have elapsed since the
    last reset (or series start).
    """
    beta = np.log(2.0) / hl_seconds
    n = len(ts_seconds)
    if n == 0:
        return {k: np.zeros(0) for k in ["lambda_buy", "lambda_sell", "imbalance", "is_warm"]}

    lam_buy = np.zeros(n)
    lam_sell = np.zeros(n)
    is_warm = np.zeros(n, dtype=bool)
    warmup_s = warmup_factor * hl_seconds
    reset_set = set(session_reset_ts.tolist()) if session_reset_ts is not None else set()

    lb = 0.0
    ls = 0.0
    warmup_start = float(ts_seconds[0])
    prev_ts = float(ts_seconds[0])

    for i in range(n):
        t = float(ts_seconds[i])
        if t in reset_set:
            lb, ls = 0.0, 0.0
            warmup_start = t
            prev_ts = t
        else:
            dt = max(t - prev_ts, 0.0)
            decay = np.exp(-beta * dt)
            lb *= decay
            ls *= decay

        lb += float(buys_volume[i])
        ls += float(sells_volume[i])

        lam_buy[i] = lb
        lam_sell[i] = ls
        is_warm[i] = (t - warmup_start) >= warmup_s
        prev_ts = t

    return {
        "lambda_buy": lam_buy,
        "lambda_sell": lam_sell,
        "imbalance": lam_buy - lam_sell,
        "is_warm": is_warm,
    }


# ---------------------------------------------------------------------------
# §8.D  CVD with dual reset (globex weekly, RTH daily) + RTH-bounded rolling
# ---------------------------------------------------------------------------

def cvd_with_dual_reset(
    bars: pl.DataFrame,
    ts_col: str = "ts",
    buys_col: str = "buys_qty",
    sells_col: str = "sells_qty",
) -> pl.DataFrame:
    """Add cvd_globex, cvd_rth, bars_since_rth_reset columns.

    RTH session = 09:30 ET through next 09:30 ET.
    Globex session = ISO week in America/New_York (V1 approximation).

    Implementation: explicit Python loop over rows. At 140k rows/5-yr × ~100 bars/day
    this is <1s and avoids polars window-function edge cases around reset detection.
    """
    # Compute ET time-of-day in minutes and the RTH-session date key.
    df = bars.with_columns(
        [
            pl.col(ts_col).dt.convert_time_zone("America/New_York").alias("_ts_et"),
            (pl.col(buys_col) - pl.col(sells_col)).alias("_signed"),
        ]
    )
    df = df.with_columns(
        [
            # IMPORTANT: polars dt.hour()/minute() return i8 which overflow at
            # hour*60 = 480 > 127. Cast to Int32 first.
            (
                pl.col("_ts_et").dt.hour().cast(pl.Int32) * 60
                + pl.col("_ts_et").dt.minute().cast(pl.Int32)
            ).alias("_min_et"),
            pl.col("_ts_et").dt.date().alias("_date_et"),
            pl.col("_ts_et").dt.iso_year().alias("_iw_year"),
            pl.col("_ts_et").dt.week().alias("_iw_week"),
        ]
    )
    # Compute the RTH-session date inside polars (robust date arithmetic),
    # then extract as int (days since epoch) for cheap comparison in the Python loop.
    df = df.with_columns(
        pl.when(pl.col("_min_et") >= 570)
        .then(pl.col("_date_et"))
        .otherwise(pl.col("_date_et") - pl.duration(days=1))
        .alias("_rth_session_date")
    )

    rth_session_int = df["_rth_session_date"].cast(pl.Int64).to_numpy()
    iw_year = df["_iw_year"].to_numpy()
    iw_week = df["_iw_week"].to_numpy()
    signed = df["_signed"].to_numpy().astype(np.float64)

    n = len(df)
    cvd_globex = np.zeros(n)
    cvd_rth = np.zeros(n)
    bars_since_rth_reset = np.zeros(n, dtype=np.int64)

    cg = 0.0
    cr = 0.0
    since_rth = 0
    prev_week_key: tuple | None = None
    prev_rth_key: int | None = None

    for i in range(n):
        rth_key = int(rth_session_int[i])
        week_key = (int(iw_year[i]), int(iw_week[i]))

        # Globex reset on new ISO week
        if prev_week_key is not None and week_key != prev_week_key:
            cg = 0.0
        # RTH reset when session key changes
        if prev_rth_key is not None and rth_key != prev_rth_key:
            cr = 0.0
            since_rth = 0

        cg += signed[i]
        cr += signed[i]
        cvd_globex[i] = cg
        cvd_rth[i] = cr
        bars_since_rth_reset[i] = since_rth
        since_rth += 1
        prev_week_key = week_key
        prev_rth_key = rth_key

    result = bars.with_columns(
        [
            pl.Series("cvd_globex", cvd_globex),
            pl.Series("cvd_rth", cvd_rth),
            pl.Series("bars_since_rth_reset", bars_since_rth_reset),
        ]
    )
    return result


def rolling_rth_bounded(
    values: np.ndarray,
    bars_since_reset: np.ndarray,
    window: int,
    min_bars: int = 5,
    agg: Literal["max", "min", "mean", "std"] = "max",
) -> np.ndarray:
    """Rolling aggregate that dynamically bounds at RTH reset per §8.D.

    `effective_window = min(window, bars_since_reset + 1)`
    If effective_window < min_bars → emit NaN.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if agg == "max":
        fn = np.nanmax
    elif agg == "min":
        fn = np.nanmin
    elif agg == "mean":
        fn = np.nanmean
    elif agg == "std":
        fn = np.nanstd
    else:
        raise ValueError(f"agg must be max/min/mean/std, got {agg!r}")

    for i in range(n):
        eff = min(int(window), int(bars_since_reset[i]) + 1)
        if eff < min_bars:
            continue
        out[i] = fn(values[i - eff + 1 : i + 1])
    return out


# ---------------------------------------------------------------------------
# §8.E  Strict-backward asof join helper (no lookahead)
# ---------------------------------------------------------------------------

def asof_strict_backward(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str,
    right_on: str,
    max_staleness_ns: int | None = None,
) -> pl.DataFrame:
    """Asof-join `right` onto `left` with strict `<` predicate (no exact-match leak).

    Implementation: shift `right` timestamps by 1 ns before the standard backward
    asof; polars' default allows equality, which is a lookahead in our context.

    If `max_staleness_ns` is provided, left rows whose nearest right match is older
    than that are set to null for joined columns.
    """
    # Shift right timestamps by -1 ns so that equality is treated as strictly older.
    right_shifted = right.with_columns(
        (pl.col(right_on) + pl.duration(nanoseconds=1)).alias(right_on)
    )
    joined = left.join_asof(
        right_shifted,
        left_on=left_on,
        right_on=right_on,
        strategy="backward",
        tolerance=(pl.duration(nanoseconds=max_staleness_ns) if max_staleness_ns else None),
    )
    return joined


# ---------------------------------------------------------------------------
# §8.F  Round-number pin distance (era-parametric)
# ---------------------------------------------------------------------------

def round_number_pin_distance(close: pl.Series | np.ndarray, N: float) -> pl.Series:
    """Distance to nearest N-point strike, symmetric V-shape: min(close%N, N−close%N).

    For N=50, close=5000 → 0; 5025 → 25; 5049.75 → 0.25; 5050.25 → 0.25.
    """
    arr = close.to_numpy() if isinstance(close, pl.Series) else np.asarray(close)
    rem = np.mod(arr, N)
    dist = np.minimum(rem, N - rem)
    return pl.Series(dist)
