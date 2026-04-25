"""Overnight-window features attached to RTH bars.

For each trading date, compute features describing what happened overnight
between yesterday's RTH close and today's RTH open. Broadcast those
per-day features to every bar of today's session so the model can use
overnight context at any point during the day.

Why this matters: overnight 23h-of-CME-trading carries a huge share of the
information used by RTH price discovery — Asian risk-off events, EU policy
news, pre-market data releases all hit overnight. A pure "RTH-only" feature
set blinds the model to all of that. With session-flag conditioning + these
overnight aggregates, the model sees the structural setup before it predicts.

Definition of overnight: from yesterday's last RTH bar (close ≈ 16:00 ET)
to today's first RTH bar (open ≈ 09:00-09:30 ET). Bars in between are the
"overnight window."

Features (per trading day, broadcast to every bar of that day):

    overnight_log_return     log(rth_open_today / rth_close_yesterday)
    overnight_realized_vol   sqrt(sum of squared 15-min log returns during overnight)
    overnight_volume_total   sum(volume) of all bars in the overnight window
    overnight_n_bars         count of bars in the overnight window

Use case: on the 11:00 ET bar (mid-RTH), the model sees overnight_log_return
= -0.5% as a feature, knows there was a sizeable overnight gap down, and
can condition predictions on that context.
"""
from __future__ import annotations

import polars as pl


def attach_overnight_features(
    bars: pl.DataFrame,
    ts_col: str = "ts",
    open_col: str = "open",
    close_col: str = "close",
    volume_col: str = "volume",
    is_rth_col: str = "is_rth",
) -> pl.DataFrame:
    """Attach overnight-window aggregates to every bar.

    Requires `is_rth` already present (call `attach_session_flags` first).

    Computes:
        overnight_log_return    : log(today_first_rth_open / yesterday_last_rth_close)
        overnight_realized_vol  : sqrt of sum(log_ret^2) over bars in overnight window
        overnight_volume_total  : total volume in overnight window
        overnight_n_bars        : count of bars in overnight window

    All four are constant within a trading date (broadcast from daily lookup).
    """
    if is_rth_col not in bars.columns:
        raise ValueError(
            f"'{is_rth_col}' not found; call attach_session_flags() first"
        )

    # Trading date — bars from ET 16:00 onward belong to the NEXT day's
    # trading session (overnight setup for tomorrow's RTH). Bars before
    # ET 16:00 belong to today. This keeps overnight bars + the next morning's
    # RTH session in a single grouping key.
    et = pl.col(ts_col).dt.convert_time_zone("US/Eastern")
    et_hour = et.dt.hour()
    et_date = et.dt.date()
    trading_date = pl.when(et_hour >= 16).then(
        et_date + pl.duration(days=1)
    ).otherwise(et_date)
    df = bars.with_columns(trading_date.alias("_trading_date"))

    # Per-trading-date: first RTH open, last RTH close, count + volume of non-RTH bars
    # (the non-RTH bars are the overnight + EU + ASIA preceding today's RTH session).
    rth_aggs = (
        df.filter(pl.col(is_rth_col) == 1)
        .group_by("_trading_date")
        .agg([
            pl.col(open_col).first().alias("_rth_open_today"),
            pl.col(close_col).last().alias("_rth_close_today"),
        ])
        .sort("_trading_date")
        .with_columns(
            pl.col("_rth_close_today").shift(1).alias("_rth_close_yesterday")
        )
    )

    # Overnight realized stats: log returns for bars BEFORE today's first RTH bar
    # but AFTER yesterday's last RTH bar.
    # We compute per-bar log-returns first, then aggregate over the overnight window.
    df = df.with_columns(
        (pl.col(close_col).log() - pl.col(close_col).shift(1).log()).alias("_log_ret")
    )

    # Overnight = bars with is_rth==0 since yesterday's last RTH bar.
    overnight = (
        df.filter(pl.col(is_rth_col) == 0)
        .group_by("_trading_date")
        .agg([
            (pl.col("_log_ret") ** 2).sum().alias("_ovnt_var_sum"),
            pl.col(volume_col).sum().alias("_ovnt_volume"),
            pl.len().alias("_ovnt_n"),
        ])
    )

    # Combine: rth_aggs has gap return; overnight has realized vol / volume.
    daily = rth_aggs.join(overnight, on="_trading_date", how="left").with_columns([
        (pl.col("_rth_open_today").log() - pl.col("_rth_close_yesterday").log())
            .alias("overnight_log_return"),
        pl.col("_ovnt_var_sum").sqrt().alias("overnight_realized_vol"),
        pl.col("_ovnt_volume").alias("overnight_volume_total"),
        pl.col("_ovnt_n").alias("overnight_n_bars"),
    ]).select([
        "_trading_date",
        "overnight_log_return",
        "overnight_realized_vol",
        "overnight_volume_total",
        "overnight_n_bars",
    ])

    return df.join(daily, on="_trading_date", how="left").drop(["_trading_date", "_log_ret"])
