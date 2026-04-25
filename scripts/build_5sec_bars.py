"""Build raw 5-sec Phase A bars (no downsample) per (instrument, day).

Mirrors `scripts/build_phase_a_bars.py` but writes the 5-sec output directly
without the 15-min downsample. Used downstream by:
  - VPIN volume-bucket aggregation (`engines.vpin_volume_buckets`)
  - Hawkes intensity recursion with actual Δt
  - Sub-bar realized moments (skew, kurt, quarticity, bipower)

Output: per-day 5-sec parquet:
    {OUT_ROOT}/{INSTR}/5s/{INSTR}_{YYYYMMDD}_5s.parquet

Idempotent: skip if output exists.

Usage:
    python scripts/build_5sec_bars.py \
        --instrument ES \
        --start 2020-01-01 --end 2024-12-31 \
        --out-root /N/project/.../bars_5sec
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars_5sec import build_5sec_bars_core
from src.data.ingest import locate, read_taq, split_trades_quotes
from src.data.roll import iter_front_series

ALL_INSTRUMENTS = [
    "ES", "NQ", "RTY", "YM",
    "6A", "6B", "6C", "6E", "6J",
    "BZ", "CL", "HO", "NG", "RB",
    "GC", "HG", "PA", "PL", "SI",
    "SR3", "TN", "ZB", "ZF", "ZN", "ZT",
    "ZC", "ZL", "ZM", "ZS", "ZW",
]


def _out_path(out_root: Path, instr: str, day: date) -> Path:
    return out_root / instr / "5s" / f"{instr}_{day:%Y%m%d}_5s.parquet"


def build_day(instr: str, expiry: str, day: date, out_path: Path,
              algoseek_root: Path | None) -> tuple[int, int]:
    cf = locate("taq", instr, expiry, day, algoseek_root=algoseek_root)
    if not cf.exists:
        raise FileNotFoundError(str(cf.path))
    taq = read_taq(cf)
    trades, quotes = split_trades_quotes(taq)
    if trades.height == 0 or len(quotes) == 0:
        return 0, 0
    bars_5s = build_5sec_bars_core(trades, quotes, root=instr, expiry=expiry, every="5s")
    if bars_5s.height == 0:
        return 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars_5s.write_parquet(out_path, compression="zstd", compression_level=3)
    return trades.height, bars_5s.height


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=ALL_INSTRUMENTS)
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--out-root", required=True)
    p.add_argument("--algoseek-root", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_root = Path(args.out_root)
    algoseek_root = Path(args.algoseek_root) if args.algoseek_root else None

    n_ok = n_skip = n_err = n_empty = 0

    for fc in iter_front_series(args.instrument, start, end, dataset="taq",
                                 algoseek_root=algoseek_root):
        out = _out_path(out_root, args.instrument, fc.day)
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            n_trades, n_bars = build_day(
                args.instrument, fc.expiry, fc.day, out, algoseek_root=algoseek_root,
            )
            if n_trades == 0:
                n_empty += 1
                print(f"[empty] {fc.day}  {fc.expiry}")
            else:
                n_ok += 1
                print(f"[ok]    {fc.day}  {fc.expiry}  trades={n_trades:,}  bars={n_bars:,}  → {out.name}")
        except Exception as e:
            n_err += 1
            print(f"[err]  {fc.day}  {fc.expiry}  {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\n[summary] {args.instrument} 5s [{start}..{end}]")
    print(f"  built={n_ok}  empty={n_empty}  skipped={n_skip}  errors={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
