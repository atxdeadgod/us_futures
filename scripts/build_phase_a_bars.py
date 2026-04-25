"""Build Phase A bars (OHLCV + aggressor + L1 close + spread sub-bar + CVD)
for one (instrument, day) at a time, downsampled to 15-min, written per-day parquet.

Phase A schema (from src/data/bars_5sec.py):
    ts, root, expiry, is_rth, is_session_warm
    OHLCV:               open, high, low, close, volume, dollar_volume
    aggressor-signed:    buys_qty, sells_qty, trades_count, unclassified_count
    implied split:       implied_volume, implied_buys, implied_sells
    L1 at close:         bid_close, ask_close, mid_close, spread_abs_close
    spread sub-bar:      spread_mean_sub, spread_std_sub, spread_max_sub, spread_min_sub
    CVD dual-reset:      cvd_globex, cvd_rth, bars_since_rth_reset

Output: per-day 15-min parquets:
    {OUT_ROOT}/{INSTR}/{HORIZON}/{INSTR}_{YYYYMMDD}_{HORIZON}.parquet

Idempotent: skip if output exists.

Usage:
    python scripts/build_phase_a_bars.py \
        --instrument ES \
        --start 2020-01-01 --end 2023-12-31 \
        --horizon 15m \
        --out-root /N/project/.../bars_phase_a
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_5sec import build_5sec_bars_core
from src.data.bars_downsample import downsample_bars
from src.data.ingest import locate, read_taq, split_trades_quotes
from src.data.roll import iter_front_series


def _out_path(out_root: Path, instr: str, horizon: str, day: date) -> Path:
    return out_root / instr / horizon / f"{instr}_{day:%Y%m%d}_{horizon}.parquet"


def build_day(
    instr: str, expiry: str, day: date, horizon: str,
    out_path: Path, algoseek_root: Path | None,
) -> tuple[int, int]:
    cf = locate("taq", instr, expiry, day, algoseek_root=algoseek_root)
    if not cf.exists:
        raise FileNotFoundError(str(cf.path))
    taq = read_taq(cf)
    trades, quotes = split_trades_quotes(taq)
    if trades.height == 0 or len(quotes) == 0:
        # Zero-trade or zero-quote days (holidays, halts) — soft skip.
        # Not an error; caller treats as no-build instead of fatal.
        return 0, 0
    bars_5s = build_5sec_bars_core(trades, quotes, root=instr, expiry=expiry, every="5s")
    if bars_5s.height == 0:
        return 0, 0  # also a soft skip
    bars = downsample_bars(bars_5s, target_every=horizon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(out_path, compression="zstd", compression_level=3)
    return trades.height, bars.height


ALL_INSTRUMENTS = [
    # Equity indices (V1 trading instruments)
    "ES", "NQ", "RTY", "YM",
    # FX (USD pairs — synthesize DXY composite)
    "6A", "6B", "6C", "6E", "6J",
    # Energy
    "BZ", "CL", "HO", "NG", "RB",
    # Metals
    "GC", "HG", "PA", "PL", "SI",
    # Rates / curve
    "SR3", "TN", "ZB", "ZF", "ZN", "ZT",
    # Ags
    "ZC", "ZL", "ZM", "ZS", "ZW",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=ALL_INSTRUMENTS)
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--horizon", default="15m")
    p.add_argument("--out-root", required=True)
    p.add_argument("--algoseek-root", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_root = Path(args.out_root)
    algoseek_root = Path(args.algoseek_root) if args.algoseek_root else None

    n_ok = n_skip = n_err = n_empty = 0
    errors: list[str] = []

    for fc in iter_front_series(args.instrument, start, end, dataset="taq",
                                 algoseek_root=algoseek_root):
        out = _out_path(out_root, args.instrument, args.horizon, fc.day)
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            n_trades, n_bars = build_day(
                args.instrument, fc.expiry, fc.day, args.horizon, out,
                algoseek_root=algoseek_root,
            )
            if n_trades == 0:
                n_empty += 1
                print(f"[empty] {fc.day}  {fc.expiry}  zero trades/bars (holiday or halt)")
            else:
                n_ok += 1
                print(f"[ok]    {fc.day}  {fc.expiry}  trades={n_trades:,}  bars={n_bars}  → {out.name}")
        except Exception as e:
            n_err += 1
            msg = f"[err]  {fc.day}  {fc.expiry}  {type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            traceback.print_exc(file=sys.stderr)

    print(f"\n[summary] {args.instrument} [{start}..{end}] horizon={args.horizon}")
    print(f"  built={n_ok}  empty(holiday)={n_empty}  skipped(already-have)={n_skip}  errors={n_err}")
    # Empty days are NOT failures (holidays, halts); they're expected.
    # Genuinely-broken files (read errors, schema mismatch) still bubble up
    # as exceptions and exit non-zero — keeps the dependency chain honest.
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
