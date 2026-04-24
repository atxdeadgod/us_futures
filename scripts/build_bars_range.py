"""Build OHLCV 15-min bars per (instrument, trading day) for label tuning.

Lean pipeline — no Phase B/C/D enrichment, no L1 stream, no CVD, no aggressor
split. Just raw OHLCV + volume sufficient for triple-barrier label tuning.

Output layout (one parquet per day):
    {OUT_ROOT}/{INSTR}/{HORIZON}/{INSTR}_{YYYYMMDD}_{HORIZON}.parquet

Idempotent: skips days whose output already exists.

Usage:
    python scripts/build_bars_range.py \
        --instrument ES \
        --start 2020-01-01 --end 2023-12-31 \
        --horizon 15m \
        --out-root /N/project/ksb-finance-backtesting/data/label_tuning/bars_ohlcv
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

from src.data.ingest import read_taq, split_trades_quotes, locate
from src.data.roll import iter_front_series


def _trades_to_ohlcv(trades: pl.DataFrame, every: str) -> pl.DataFrame:
    """Aggregate trades into OHLCV bars at `every` horizon. Trades-only; no quote fill."""
    return (
        trades.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="right")
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("quantity").sum().alias("volume"),
                (pl.col("price") * pl.col("quantity")).sum().alias("dollar_volume"),
                pl.len().alias("trades_count"),
            ]
        )
    )


def _out_path(out_root: Path, instr: str, horizon: str, day: date) -> Path:
    return out_root / instr / horizon / f"{instr}_{day:%Y%m%d}_{horizon}.parquet"


def build_day(
    instr: str,
    expiry: str,
    day: date,
    horizon: str,
    out_path: Path,
    algoseek_root: Path | None = None,
) -> tuple[int, int]:
    """Build one day's OHLCV bars for (instr, expiry). Returns (n_trades, n_bars).

    Raises on ingest failure so the caller can log + skip.
    """
    cf = locate("taq", instr, expiry, day, algoseek_root=algoseek_root)
    if not cf.exists:
        raise FileNotFoundError(str(cf.path))
    taq = read_taq(cf)
    trades, _quotes = split_trades_quotes(taq)
    if trades.height == 0:
        raise ValueError(f"zero trades in {cf.path}")
    bars = _trades_to_ohlcv(trades, every=horizon)
    bars = bars.with_columns(
        [
            pl.lit(instr).alias("root"),
            pl.lit(expiry).alias("expiry"),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(out_path, compression="zstd", compression_level=3)
    return trades.height, bars.height


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--horizon", default="15m")
    p.add_argument("--out-root", required=True,
                   help="Base directory; day files go under {out}/{instr}/{horizon}/")
    p.add_argument("--algoseek-root", default=None,
                   help="Override $ALGOSEEK_ROOT (default: HPC project path)")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild days even if output parquet already exists")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_root = Path(args.out_root)
    algoseek_root = Path(args.algoseek_root) if args.algoseek_root else None

    n_ok = n_skip_exists = n_err = n_no_front = 0
    errors: list[str] = []

    for fc in iter_front_series(
        args.instrument, start, end, dataset="taq", algoseek_root=algoseek_root
    ):
        out = _out_path(out_root, args.instrument, args.horizon, fc.day)
        if out.exists() and not args.overwrite:
            n_skip_exists += 1
            continue
        try:
            n_trades, n_bars = build_day(
                args.instrument, fc.expiry, fc.day, args.horizon, out,
                algoseek_root=algoseek_root,
            )
            n_ok += 1
            print(f"[ok]   {fc.day}  {fc.expiry}  trades={n_trades:,}  bars={n_bars}  → {out.name}")
        except Exception as e:
            n_err += 1
            msg = f"[err]  {fc.day}  {fc.expiry}  {type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            traceback.print_exc(file=sys.stderr)

    # Also count days in [start,end] where no front-month existed (holidays / missing data)
    d = start
    while d <= end:
        if d.weekday() < 5:
            n_no_front += 1
        d += timedelta(days=1)
    n_no_front -= (n_ok + n_skip_exists + n_err)  # expected weekdays minus resolved

    print(f"\n[summary] {args.instrument} [{start}..{end}] horizon={args.horizon}")
    print(f"  built     : {n_ok}")
    print(f"  skipped   : {n_skip_exists}  (already present)")
    print(f"  errors    : {n_err}")
    print(f"  no-front  : {max(n_no_front, 0)}  (holiday / missing data)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
