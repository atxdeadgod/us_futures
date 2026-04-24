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
# ATR helper (true-range rolling mean)
# ---------------------------------------------------------------------------

def atr_column(
    high_col: pl.Expr, low_col: pl.Expr, close_col: pl.Expr, window: int = 20
) -> pl.Expr:
    """True-Range rolling mean. TR = max(H−L, |H−prev_C|, |L−prev_C|)."""
    prev_close = close_col.shift(1)
    tr = pl.max_horizontal(
        high_col - low_col,
        (high_col - prev_close).abs(),
        (low_col - prev_close).abs(),
    )
    return tr.rolling_mean(window_size=window)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (labels, hit_bar_offset, realized_return).

    hit_bar_offset: bars from label_row until barrier hit (or T if time-expired).
    realized_return: log-return from close[i] to hit price (exact barrier price for
        ±1 labels; close[i+T] for 0 labels).
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    offsets = np.zeros(n, dtype=np.int32)
    rets = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if np.isnan(atr[i]):
            labels[i] = 0
            offsets[i] = 0
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
            continue

        if hit_up != -1 and hit_dn == -1:
            labels[i] = 1
            offsets[i] = hit_up - i
            rets[i] = np.log(upper / close[i])
        elif hit_dn != -1 and hit_up == -1:
            labels[i] = -1
            offsets[i] = hit_dn - i
            rets[i] = np.log(lower / close[i])
        else:
            # Both hit — same bar or different bars
            if hit_up < hit_dn:
                labels[i] = 1
                offsets[i] = hit_up - i
                rets[i] = np.log(upper / close[i])
            elif hit_dn < hit_up:
                labels[i] = -1
                offsets[i] = hit_dn - i
                rets[i] = np.log(lower / close[i])
            else:
                # Same bar j: use open-vs-close direction
                j = hit_up
                if close[j] > open_[j]:
                    labels[i] = 1
                    offsets[i] = j - i
                    rets[i] = np.log(upper / close[i])
                else:
                    labels[i] = -1
                    offsets[i] = j - i
                    rets[i] = np.log(lower / close[i])
    return labels, offsets, rets


def triple_barrier_labels(
    bars: pl.DataFrame,
    k_up: float = 1.5,
    k_dn: float = 1.0,
    T: int = 8,
    atr_window: int = 20,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Emit triple-barrier labels on a bar frame.

    Returns the original frame with appended columns:
        atr, label (Int8 in {-1,0,+1}), hit_offset (Int32), realized_ret (Float64).
    """
    df = bars.with_columns(
        atr_column(pl.col(high_col), pl.col(low_col), pl.col(close_col), window=atr_window).alias("atr")
    )
    close = df[close_col].to_numpy()
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    open_ = df[open_col].to_numpy()
    atr = df["atr"].to_numpy()

    labels, offsets, rets = _triple_barrier_np(close, high, low, open_, atr, k_up, k_dn, T)

    return df.with_columns(
        [
            pl.Series("label", labels, dtype=pl.Int8),
            pl.Series("hit_offset", offsets, dtype=pl.Int32),
            pl.Series("realized_ret", rets, dtype=pl.Float64),
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
    mean_ret_pos: float
    mean_ret_neg: float
    mean_ret_zero: float
    mean_hit_offset_pos: float
    mean_hit_offset_neg: float
    mean_hit_offset_zero: float
    n_total: int
    label_forward_return_corr: float  # spearman-ish; here we use simple pearson label × fwd-return
    balance_score: float  # 0..1, higher = more balanced class distribution


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
                    valid = labeled.filter(
                        pl.col("atr").is_not_null() & pl.col("realized_ret").is_not_null()
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

                    rows.append(
                        LabelStats(
                            k_up=k_up, k_dn=k_dn, T=T, atr_window=atr_w,
                            frac_pos=frac_pos, frac_neg=frac_neg, frac_zero=frac_zero,
                            mean_ret_pos=_mean(1), mean_ret_neg=_mean(-1), mean_ret_zero=_mean(0),
                            mean_hit_offset_pos=_mean_offset(1),
                            mean_hit_offset_neg=_mean_offset(-1),
                            mean_hit_offset_zero=_mean_offset(0),
                            n_total=n,
                            label_forward_return_corr=corr,
                            balance_score=_balance_score(frac_pos, frac_neg, frac_zero),
                        )
                    )
    return pl.DataFrame([vars(r) for r in rows])
