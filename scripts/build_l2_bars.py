"""Track B: enrich Phase A bars with L1-L10 book snapshots from depth (MBP-10).

Consumes:
    Phase A 15-min bars at {BARS_PHASE_A_ROOT}/{INSTR}/15m/{INSTR}_{YYYYMMDD}_15m.parquet
    Algoseek depth files at {ALGOSEEK_ROOT}/depth/{INSTR}/{YYYY}/{YYYYMMDD}/{EXPIRY}.csv.gz

Produces (Phase A + Phase B):
    {OUT_ROOT}/{INSTR}/15m/{INSTR}_{YYYYMMDD}_15m.parquet

The output adds 60 columns (bid_px_L1..L10, bid_sz_L1..L10, bid_ord_L1..L10
plus ask side) + `book_ts_close` to each Phase A bar.

Idempotent: skip if output exists.

Usage:
    python scripts/build_l2_bars.py \
        --instrument ES \
        --bars-phase-a-root /N/project/.../bars_phase_a \
        --start 2020-01-01 --end 2024-12-31 \
        --out-root /N/project/.../bars_phase_ab
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

from src.data.depth_snap import attach_book_snapshot
from src.data.ingest import locate, read_depth
from src.data.roll import iter_front_series


def _phase_a_path(root: Path, instr: str, day: date) -> Path:
    return root / instr / "15m" / f"{instr}_{day:%Y%m%d}_15m.parquet"


def _out_path(out_root: Path, instr: str, day: date) -> Path:
    return out_root / instr / "15m" / f"{instr}_{day:%Y%m%d}_15m.parquet"


def build_day(
    instr: str, expiry: str, day: date,
    bars_phase_a_root: Path, out_path: Path,
    algoseek_root: Path | None,
) -> tuple[int, int]:
    pa_path = _phase_a_path(bars_phase_a_root, instr, day)
    if not pa_path.exists():
        raise FileNotFoundError(f"Phase A bars missing: {pa_path}")
    bars = pl.read_parquet(pa_path)
    cf = locate("depth", instr, expiry, day, algoseek_root=algoseek_root)
    if not cf.exists:
        raise FileNotFoundError(f"depth file missing: {cf.path}")
    depth = read_depth(cf)
    bars_l2 = attach_book_snapshot(bars, depth, only_regular=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars_l2.write_parquet(out_path, compression="zstd", compression_level=3)
    return depth.height, bars_l2.height


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True, choices=["ES", "NQ", "RTY", "YM"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--bars-phase-a-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--algoseek-root", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    pa_root = Path(args.bars_phase_a_root)
    out_root = Path(args.out_root)
    algoseek_root = Path(args.algoseek_root) if args.algoseek_root else None

    n_ok = n_skip = n_err = n_no_pa = 0
    errors: list[str] = []

    for fc in iter_front_series(args.instrument, start, end, dataset="depth",
                                 algoseek_root=algoseek_root):
        out = _out_path(out_root, args.instrument, fc.day)
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        if not _phase_a_path(pa_root, args.instrument, fc.day).exists():
            n_no_pa += 1
            continue
        try:
            n_dep, n_bars = build_day(
                args.instrument, fc.expiry, fc.day,
                pa_root, out, algoseek_root=algoseek_root,
            )
            n_ok += 1
            print(f"[ok]   {fc.day}  {fc.expiry}  depth_events={n_dep:,}  bars={n_bars}  → {out.name}")
        except Exception as e:
            n_err += 1
            msg = f"[err]  {fc.day}  {fc.expiry}  {type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            traceback.print_exc(file=sys.stderr)

    print(f"\n[summary] {args.instrument} [{start}..{end}]")
    print(f"  built={n_ok}  skipped={n_skip}  errors={n_err}  no_phase_a={n_no_pa}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
