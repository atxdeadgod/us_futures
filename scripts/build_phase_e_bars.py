"""Phase E: per-bar execution-quality + cancel-proxy + quote-event aggregates.

Consumes raw TAQ + depth and emits a per-day 15-min parquet with Phase E
columns only. Downstream `single_contract.attach_phase_e_features` asof-joins
this onto the futures panel.

Output schema (15-min bars):
    ts
    eff_spread_sum, eff_spread_weight, eff_spread_count             (T1.35)
    eff_spread_buy_sum, eff_spread_buy_weight                        (T1.36 build)
    eff_spread_sell_sum, eff_spread_sell_weight                      (T1.37 build)
    n_large_trades, large_trade_volume                               (T1.23)
    hidden_absorption_volume, hidden_absorption_trades               (T1.47, T7.12)
    net_bid_decrement_no_trade_L1, net_ask_decrement_no_trade_L1     (T1.43)
    quote_update_count                                                (T1.24)

Idempotent: skip if output exists.

Usage:
    python scripts/build_phase_e_bars.py \\
        --instrument ES \\
        --start 2020-01-01 --end 2024-12-31 \\
        --out-root /N/project/.../bars_phase_e \\
        --algoseek-root /N/project/.../algoseek_futures
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_5sec import BAR_EVERY
from src.data.bars_cancel import cancel_proxy_bars
from src.data.bars_exec import (
    effective_spread_bars,
    hidden_absorption_bars,
    large_trade_bars,
    quote_direction_bars,
)
from src.data.ingest import locate, read_depth, read_taq, split_trades_quotes
from src.data.roll import iter_front_series


# Per-column aggregation rules for the 5s→15m downsample. Phase E columns are
# almost all sums (counts/volumes); the eff_spread_*_weight cols are also sums
# (so the ratio sum_at_15m / weight_at_15m gives the volume-weighted average
# spread for that 15m window).
_PHASE_E_AGG_RULES: dict[str, str] = {
    "eff_spread_sum": "sum", "eff_spread_weight": "sum", "eff_spread_count": "sum",
    "eff_spread_buy_sum": "sum", "eff_spread_buy_weight": "sum",
    "eff_spread_sell_sum": "sum", "eff_spread_sell_weight": "sum",
    "n_large_trades": "sum", "large_trade_volume": "sum",
    "hidden_absorption_volume": "sum", "hidden_absorption_trades": "sum",
    "net_bid_decrement_no_trade_L1": "sum", "net_ask_decrement_no_trade_L1": "sum",
    # T1.28 side-conditioned cancel-proxy extensions
    "bid_sz_L1_delta_signed": "sum",  # signed bar deltas; sum of 5s deltas = full-bar delta
    "ask_sz_L1_delta_signed": "sum",
    "hit_bid_vol": "sum",  # sell-aggressor volume
    "lift_ask_vol": "sum",  # buy-aggressor volume
    "quote_update_count": "sum",
    # T1.25 quote-direction event counts
    "bid_up_count": "sum", "bid_down_count": "sum",
    "ask_up_count": "sum", "ask_down_count": "sum",
}


def _quote_event_count_bars(quotes: pl.DataFrame, every: str = BAR_EVERY) -> pl.DataFrame:
    """Count quote events (bid + ask updates) per 5-sec bar — supports T1.24 quote/trade ratio."""
    return (
        quotes.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(pl.len().alias("quote_update_count"))
    )


def _downsample_phase_e(bars_5s: pl.DataFrame, target_every: str = "15m") -> pl.DataFrame:
    """Aggregate 5s Phase E columns to the target horizon using sum rules."""
    aggs = []
    for col in bars_5s.columns:
        if col == "ts":
            continue
        rule = _PHASE_E_AGG_RULES.get(col, "sum")
        if rule == "sum":
            aggs.append(pl.col(col).sum().alias(col))
        elif rule == "last":
            aggs.append(pl.col(col).last().alias(col))
        else:
            raise ValueError(f"unhandled rule {rule!r} for {col!r}")
    return (
        bars_5s.sort("ts")
        .group_by_dynamic("ts", every=target_every, closed="left", label="right")
        .agg(aggs)
    )


def _out_path(out_root: Path, instr: str, day: date, horizon: str) -> Path:
    return out_root / instr / horizon / f"{instr}_{day:%Y%m%d}_{horizon}.parquet"


def build_day(
    instr: str, expiry: str, day: date,
    out_path: Path, algoseek_root: Path | None,
    target_every: str = "15m",
) -> tuple[int, int]:
    taq_cf = locate("taq", instr, expiry, day, algoseek_root=algoseek_root)
    depth_cf = locate("depth", instr, expiry, day, algoseek_root=algoseek_root)
    if not taq_cf.exists:
        raise FileNotFoundError(f"taq missing: {taq_cf.path}")
    if not depth_cf.exists:
        raise FileNotFoundError(f"depth missing: {depth_cf.path}")

    taq = read_taq(taq_cf)
    trades, quotes = split_trades_quotes(taq)
    depth = read_depth(depth_cf)

    if trades.height == 0 or quotes.height == 0:
        return 0, 0  # soft skip — holiday or zero-trade day

    eff = effective_spread_bars(trades, quotes, every=BAR_EVERY)
    large = large_trade_bars(trades, every=BAR_EVERY, threshold_pct=0.99)
    hidden = hidden_absorption_bars(trades, depth, every=BAR_EVERY, only_regular=True)
    cancel = cancel_proxy_bars(trades, depth, every=BAR_EVERY, only_regular=True)
    qcount = _quote_event_count_bars(quotes, every=BAR_EVERY)
    qdir = quote_direction_bars(quotes, every=BAR_EVERY)

    # Outer-join all six frames on ts (5-sec resolution).
    bars_5s = eff.join(large, on="ts", how="full", coalesce=True)
    bars_5s = bars_5s.join(hidden, on="ts", how="full", coalesce=True)
    bars_5s = bars_5s.join(cancel, on="ts", how="full", coalesce=True)
    bars_5s = bars_5s.join(qcount, on="ts", how="full", coalesce=True)
    bars_5s = bars_5s.join(qdir, on="ts", how="full", coalesce=True).sort("ts")

    # Fill nulls (bars with no event of that kind get 0).
    for col in bars_5s.columns:
        if col == "ts":
            continue
        bars_5s = bars_5s.with_columns(pl.col(col).fill_null(0))

    bars_15m = _downsample_phase_e(bars_5s, target_every=target_every)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars_15m.write_parquet(out_path, compression="zstd", compression_level=3)
    return trades.height, bars_15m.height


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--out-root", required=True)
    p.add_argument("--algoseek-root", default=None)
    p.add_argument("--horizon", default="15m")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_root = Path(args.out_root)
    algoseek_root = Path(args.algoseek_root) if args.algoseek_root else None

    n_ok = n_skip = n_err = n_empty = 0
    errors: list[str] = []

    # Iterate over the front series via TAQ (depth follows the same roll).
    for fc in iter_front_series(args.instrument, start, end, dataset="taq",
                                 algoseek_root=algoseek_root):
        out = _out_path(out_root, args.instrument, fc.day, args.horizon)
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            n_trades, n_bars = build_day(
                args.instrument, fc.expiry, fc.day, out,
                algoseek_root=algoseek_root, target_every=args.horizon,
            )
            if n_trades == 0:
                n_empty += 1
                print(f"[empty] {fc.day}  {fc.expiry}  zero trades")
            else:
                n_ok += 1
                print(f"[ok]    {fc.day}  {fc.expiry}  trades={n_trades:,}  bars={n_bars}  → {out.name}")
        except Exception as e:
            n_err += 1
            msg = f"[err]  {fc.day}  {fc.expiry}  {type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            traceback.print_exc(file=sys.stderr)

    print(f"\n[summary] {args.instrument} Phase E [{start}..{end}] horizon={args.horizon}")
    print(f"  built={n_ok}  empty={n_empty}  skipped={n_skip}  errors={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
