"""Triple-barrier labels (AFML Ch. 3) + tuning harness.

Given a bar frame with (ts, open, high, low, close, atr), compute labels
{-1, 0, +1} per bar:

  +1  upper barrier (close + k_up * atr) hit first within horizon T bars
  -1  lower barrier (close - k_dn * atr) hit first within horizon T bars
   0  neither hit — time-expired at vertical barrier T

Within-bar ambiguity (both upper and lower hit in the same bar j):
    Resolved by bar direction — if close[j] > open[j], assume up-first; else
    down-first. ~1% of bars; doesn't materially shift the label distribution.

Tuning workflow: label distribution and statistics vary substantially with
(k_up, k_dn, T, atr_window). The `tune_triple_barrier` function runs a grid
and reports class balance, mean forward return per class, and overall
information content — pick the combo with clean balance + meaningful per-class
separation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# ATR helpers
# ---------------------------------------------------------------------------

def atr_column(
    high_col: pl.Expr, low_col: pl.Expr, close_col: pl.Expr, window: int = 20
) -> pl.Expr:
    """Calendar-window ATR. TR = max(H−L, |H−prev_C|, |L−prev_C|), rolling-mean
    over the previous `window` bars regardless of time of day.

    Tends to over-size barriers in low-vol hours and under-size them at
    high-vol hours. Use `atr_time_conditional` to fix that for off-hours
    labeling.
    """
    prev_close = close_col.shift(1)
    tr = pl.max_horizontal(
        high_col - low_col,
        (high_col - prev_close).abs(),
        (low_col - prev_close).abs(),
    )
    return tr.rolling_mean(window_size=window)


def attach_atr_time_conditional(
    bars: pl.DataFrame,
    lookback_days: int = 30,
    bar_minutes: int = 15,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    ts_col: str = "ts",
    out_col: str = "atr",
) -> pl.DataFrame:
    """Attach a time-conditional ATR column to the bar frame.

    ATR = rolling mean of TR over the same bar-of-day across the last
    `lookback_days` trading days. The bar at (date d, bar-of-day b) gets
    ATR_tc = mean(TR over the (b)-bars from days d-lookback_days..d-1).

    Sizes barriers to the local intraday vol regime, fixing calendar-ATR
    mis-sizing at low-volume off-hours and high-volume RTH open.

    Note: `polars.Expr.rolling_mean(...).over(<expression>)` does not partition
    correctly in current polars versions; we materialize bar_of_day as an
    explicit column and use `.over("bar_of_day")` (column name).
    """
    bars_per_hour = 60 // bar_minutes
    et = pl.col(ts_col).dt.convert_time_zone("US/Eastern")
    df = bars.with_columns(
        (et.dt.hour() * bars_per_hour + et.dt.minute() // bar_minutes).alias("_bar_of_day")
    )
    prev_close = pl.col(close_col).shift(1)
    df = df.with_columns(
        pl.max_horizontal(
            pl.col(high_col) - pl.col(low_col),
            (pl.col(high_col) - prev_close).abs(),
            (pl.col(low_col) - prev_close).abs(),
        ).alias("_tr")
    )
    df = df.with_columns(
        pl.col("_tr").rolling_mean(window_size=lookback_days).over("_bar_of_day").alias(out_col)
    )
    return df.drop(["_bar_of_day", "_tr"])


# ---------------------------------------------------------------------------
# Core labeling (numpy loop — ~O(n·T); fast enough at n=500k, T=12)
# ---------------------------------------------------------------------------

def _triple_barrier_np(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    k_up: float,
    k_dn: float,
    T: int,
    ts_seconds: np.ndarray | None = None,
    halt_gap_seconds: int = 1800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (labels, hit_bar_offset, realized_log_return, realized_ret_pts).

    hit_bar_offset: bars from label_row until barrier hit (or T if time-expired).
    realized_log_return: log(barrier_price / close[i]) for ±1; log(close[i+T]/close[i]) for 0.
    realized_ret_pts: absolute price diff (barrier_price − close[i]) in instrument
        price units — i.e., ES points, NQ points, etc. Used downstream to evaluate
        whether a vol-scaled barrier is tradeable after costs.

    If ts_seconds is provided, bars whose forward T-window crosses a halt
    (consecutive ts gap > halt_gap_seconds, default 30min) are marked
    unlabelable: realized_ret stays NaN so the row is filtered downstream.
    Without ts_seconds, halt detection is skipped (legacy behavior).
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    offsets = np.zeros(n, dtype=np.int32)
    rets = np.full(n, np.nan, dtype=np.float64)
    rets_pts = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if np.isnan(atr[i]):
            labels[i] = 0
            offsets[i] = 0
            continue

        # Halt-aware: if the forward T-window crosses any halt boundary, drop this bar.
        if ts_seconds is not None:
            forward_end = min(i + T + 1, n)
            crosses_halt = False
            for j in range(i + 1, forward_end):
                if ts_seconds[j] - ts_seconds[j - 1] > halt_gap_seconds:
                    crosses_halt = True
                    break
            if crosses_halt:
                # leave label=0, ret=NaN — downstream filter drops it
                offsets[i] = -1
                continue

        upper = close[i] + k_up * atr[i]
        lower = close[i] - k_dn * atr[i]
        j_max = min(i + T + 1, n)

        hit_up = -1
        hit_dn = -1
        for j in range(i + 1, j_max):
            if hit_up == -1 and high[j] >= upper:
                hit_up = j
            if hit_dn == -1 and low[j] <= lower:
                hit_dn = j
            if hit_up != -1 and hit_dn != -1:
                break

        if hit_up == -1 and hit_dn == -1:
            # Time-expired
            if j_max > i + 1:
                labels[i] = 0
                offsets[i] = j_max - 1 - i
                rets[i] = np.log(close[j_max - 1] / close[i])
                rets_pts[i] = close[j_max - 1] - close[i]
            continue

        if hit_up != -1 and hit_dn == -1:
            labels[i] = 1
            offsets[i] = hit_up - i
            rets[i] = np.log(upper / close[i])
            rets_pts[i] = upper - close[i]
        elif hit_dn != -1 and hit_up == -1:
            labels[i] = -1
            offsets[i] = hit_dn - i
            rets[i] = np.log(lower / close[i])
            rets_pts[i] = lower - close[i]
        else:
            # Both hit — same bar or different bars
            if hit_up < hit_dn:
                labels[i] = 1
                offsets[i] = hit_up - i
                rets[i] = np.log(upper / close[i])
                rets_pts[i] = upper - close[i]
            elif hit_dn < hit_up:
                labels[i] = -1
                offsets[i] = hit_dn - i
                rets[i] = np.log(lower / close[i])
                rets_pts[i] = lower - close[i]
            else:
                # Same bar j: use open-vs-close direction
                j = hit_up
                if close[j] > open_[j]:
                    labels[i] = 1
                    offsets[i] = j - i
                    rets[i] = np.log(upper / close[i])
                    rets_pts[i] = upper - close[i]
                else:
                    labels[i] = -1
                    offsets[i] = j - i
                    rets[i] = np.log(lower / close[i])
                    rets_pts[i] = lower - close[i]
    return labels, offsets, rets, rets_pts


def triple_barrier_labels(
    bars: pl.DataFrame,
    k_up: float = 1.5,
    k_dn: float = 1.0,
    T: int = 8,
    atr_window: int = 20,
    atr_mode: str = "calendar",
    lookback_days: int = 30,
    bar_minutes: int = 15,
    halt_aware: bool = True,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    ts_col: str = "ts",
) -> pl.DataFrame:
    """Emit triple-barrier labels on a bar frame.

    Args:
        atr_mode: "calendar" (default — ATR over last `atr_window` bars regardless
            of time of day) or "time_conditional" (ATR over same bar-of-day across
            last `lookback_days` trading days). The TC mode fixes barrier
            mis-sizing across the 23h CME session.
        lookback_days: lookback for time_conditional ATR (only used when
            atr_mode='time_conditional').
        bar_minutes: bar duration in minutes (15 for 15-min bars). Defines the
            bar-of-day index for TC ATR.
        halt_aware: if True (default), bars whose forward T-window crosses a
            CME halt (>30min ts gap) are marked unlabelable (label=0, ret=NaN,
            offset=-1). Filter `realized_ret.is_finite()` to drop them.

    Returns the original frame with appended columns:
        atr, label (Int8 in {-1,0,+1}), hit_offset (Int32; -1 = halt-dropped),
        realized_ret (Float64, log return), realized_ret_pts (Float64, price pts).
    """
    if atr_mode == "calendar":
        df = bars.with_columns(
            atr_column(
                pl.col(high_col), pl.col(low_col), pl.col(close_col), window=atr_window
            ).alias("atr")
        )
    elif atr_mode == "time_conditional":
        df = attach_atr_time_conditional(
            bars,
            lookback_days=lookback_days,
            bar_minutes=bar_minutes,
            high_col=high_col, low_col=low_col, close_col=close_col, ts_col=ts_col,
            out_col="atr",
        )
    else:
        raise ValueError(f"atr_mode must be 'calendar' or 'time_conditional'; got {atr_mode!r}")
    close = df[close_col].to_numpy()
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    open_ = df[open_col].to_numpy()
    atr = df["atr"].to_numpy()
    ts_seconds = (
        df[ts_col].dt.epoch(time_unit="s").to_numpy().astype(np.int64) if halt_aware else None
    )

    labels, offsets, rets, rets_pts = _triple_barrier_np(
        close, high, low, open_, atr, k_up, k_dn, T, ts_seconds=ts_seconds
    )

    return df.with_columns(
        [
            pl.Series("label", labels, dtype=pl.Int8),
            pl.Series("hit_offset", offsets, dtype=pl.Int32),
            pl.Series("realized_ret", rets, dtype=pl.Float64),
            pl.Series("realized_ret_pts", rets_pts, dtype=pl.Float64),
        ]
    )


# ---------------------------------------------------------------------------
# Tuning harness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelStats:
    k_up: float
    k_dn: float
    T: int
    atr_window: int
    frac_pos: float
    frac_neg: float
    frac_zero: float
    # Log returns per class (what the model's classifier head effectively targets)
    mean_ret_pos: float
    mean_ret_neg: float
    mean_ret_zero: float
    # Absolute price diff per class (instrument price units, e.g. ES/NQ points)
    mean_ret_pts_pos: float
    mean_ret_pts_neg: float
    mean_ret_pts_zero: float
    # Same scaled to bps of entry price (regime-invariant tradeability metric)
    mean_ret_bps_pos: float
    mean_ret_bps_neg: float
    mean_ret_bps_zero: float
    # Tradeability ratio: mean |ret| in points / round-trip cost in points.
    # > 2 → comfortably tradeable; ~ 1 → break-even at best; < 1 → economically dead.
    pts_over_cost_pos: float
    pts_over_cost_neg: float
    mean_hit_offset_pos: float
    mean_hit_offset_neg: float
    mean_hit_offset_zero: float
    n_total: int
    label_forward_return_corr: float  # pearson label × fwd-return; sanity check
    balance_score: float  # 0..1, higher = more balanced class distribution


# Default round-trip cost in contract price points (spread + commission + slippage).
# Informed estimates for mid-liquid front-month futures at retail commission ~$0.85/RT.
# Override via tune_triple_barrier(..., cost_pts=...) at call site per-instrument.
DEFAULT_COST_PTS = {
    "ES": 0.50,   # 1 tick spread (0.25) + ~0.25 pt slippage+commission
    "NQ": 1.50,   # 1 tick spread (0.25) + wider slippage at higher price
    "RTY": 0.30,  # tick 0.10
    "YM": 3.00,   # tick 1.0
}


def _balance_score(p_pos: float, p_neg: float, p_zero: float) -> float:
    """Higher when closer to uniform 1/3 split; range [0, 1]."""
    # KL divergence from uniform, transformed
    probs = [p_pos, p_neg, p_zero]
    # entropy
    h = -sum(p * np.log(p + 1e-12) for p in probs)
    # max entropy for 3 classes = log(3) ≈ 1.0986
    return float(h / np.log(3))


def tune_triple_barrier(
    bars: pl.DataFrame,
    k_up_grid: list[float] | tuple[float, ...] = (1.0, 1.5, 2.0),
    k_dn_grid: list[float] | tuple[float, ...] = (1.0, 1.5, 2.0),
    T_grid: list[int] | tuple[int, ...] = (4, 8, 12),
    atr_window_grid: list[int] | tuple[int, ...] = (20,),
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    cost_pts: float = 0.5,
) -> pl.DataFrame:
    """Grid-search triple-barrier parameters on a bar frame.

    Returns a DataFrame with one row per (k_up, k_dn, T, atr_window) combo,
    reporting label-balance, per-class mean realized return, and balance score.

    Caller inspects the result DataFrame and picks the combo that trades off
    class balance × per-class mean-return separation × sample size.
    """
    rows = []
    for atr_w in atr_window_grid:
        for T in T_grid:
            for k_up in k_up_grid:
                for k_dn in k_dn_grid:
                    labeled = triple_barrier_labels(
                        bars,
                        k_up=k_up,
                        k_dn=k_dn,
                        T=T,
                        atr_window=atr_w,
                        open_col=open_col,
                        high_col=high_col,
                        low_col=low_col,
                        close_col=close_col,
                    )
                    # is_finite() catches both null AND NaN; the labeler emits NaN
                    # (not null) for warmup rows + tail rows where j_max == i+1.
                    valid = labeled.filter(
                        pl.col("atr").is_finite() & pl.col("realized_ret").is_finite()
                    )
                    n = valid.height
                    if n == 0:
                        continue
                    counts = valid.group_by("label").agg(pl.len().alias("cnt")).sort("label")
                    count_map = {int(r["label"]): int(r["cnt"]) for r in counts.iter_rows(named=True)}
                    n_pos = count_map.get(1, 0)
                    n_neg = count_map.get(-1, 0)
                    n_zero = count_map.get(0, 0)
                    frac_pos = n_pos / n
                    frac_neg = n_neg / n
                    frac_zero = n_zero / n

                    def _mean(lbl: int) -> float:
                        sub = valid.filter(pl.col("label") == lbl)["realized_ret"]
                        return float(sub.mean()) if sub.len() > 0 else float("nan")

                    def _mean_pts(lbl: int) -> float:
                        sub = valid.filter(pl.col("label") == lbl)["realized_ret_pts"]
                        return float(sub.mean()) if sub.len() > 0 else float("nan")

                    def _mean_bps(lbl: int) -> float:
                        sub = valid.filter(pl.col("label") == lbl).select(
                            (pl.col("realized_ret_pts") / pl.col(close_col) * 10_000).alias("bps")
                        )["bps"]
                        return float(sub.mean()) if sub.len() > 0 else float("nan")

                    def _mean_offset(lbl: int) -> float:
                        sub = valid.filter(pl.col("label") == lbl)["hit_offset"]
                        return float(sub.mean()) if sub.len() > 0 else float("nan")

                    # Label-to-realized-return correlation (pearson; sanity check)
                    lbl_arr = valid["label"].cast(pl.Float64).to_numpy()
                    ret_arr = valid["realized_ret"].to_numpy()
                    if np.std(lbl_arr) > 0 and np.std(ret_arr) > 0:
                        corr = float(np.corrcoef(lbl_arr, ret_arr)[0, 1])
                    else:
                        corr = float("nan")

                    pts_pos = _mean_pts(1)
                    pts_neg = _mean_pts(-1)
                    rows.append(
                        LabelStats(
                            k_up=k_up, k_dn=k_dn, T=T, atr_window=atr_w,
                            frac_pos=frac_pos, frac_neg=frac_neg, frac_zero=frac_zero,
                            mean_ret_pos=_mean(1), mean_ret_neg=_mean(-1), mean_ret_zero=_mean(0),
                            mean_ret_pts_pos=pts_pos, mean_ret_pts_neg=pts_neg,
                            mean_ret_pts_zero=_mean_pts(0),
                            mean_ret_bps_pos=_mean_bps(1), mean_ret_bps_neg=_mean_bps(-1),
                            mean_ret_bps_zero=_mean_bps(0),
                            pts_over_cost_pos=(abs(pts_pos) / cost_pts) if cost_pts > 0 and not np.isnan(pts_pos) else float("nan"),
                            pts_over_cost_neg=(abs(pts_neg) / cost_pts) if cost_pts > 0 and not np.isnan(pts_neg) else float("nan"),
                            mean_hit_offset_pos=_mean_offset(1),
                            mean_hit_offset_neg=_mean_offset(-1),
                            mean_hit_offset_zero=_mean_offset(0),
                            n_total=n,
                            label_forward_return_corr=corr,
                            balance_score=_balance_score(frac_pos, frac_neg, frac_zero),
                        )
                    )
    return pl.DataFrame([vars(r) for r in rows])
