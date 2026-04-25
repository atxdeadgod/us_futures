"""Build VX1/VX2/VX3 Phase A bars from raw VIX TAQ.

Each day we identify the front-three monthly contracts (calendar-ordered) and
build Phase A 15-min bars per slot, written one file per (slot, day):

    {OUT_ROOT}/VX{i}/15m/VX{i}_{YYYYMMDD}_15m.parquet     i in {1,2,3}

The Phase A schema is identical to futures Phase A (see `src/data/bars_5sec.py`).
We add an `expiry` column carrying the actual VX contract symbol so downstream
callers can detect rolls and apply offsets if needed.

Idempotent: skips per-(slot,day) if file already exists.

Usage:
    python scripts/build_vx_bars.py \
        --start 2020-01-01 --end 2024-12-31 \
        --horizon 15m \
        --out-root /N/project/.../bars_phase_a \
        --algoseek-root /N/project/.../algoseek_futures
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
from src.data.bars_downsample import downsample_bars
from src.data.ingest import locate, read_taq, split_trades_quotes
from src.data.roll import front_n


def _out_path(out_root: Path, slot: str, horizon: str, day: date) -> Path:
    return out_root / slot / horizon / f"{slot}_{day:%Y%m%d}_{horizon}.parquet"


def build_slot_day(
    slot: str, expiry: str, day: date, horizon: str,
    out_path: Path, algoseek_root: Path | None,
) -> tuple[int, int]:
    cf = locate("vix", "VX", expiry, day, algoseek_root=algoseek_root)
    if not cf.exists:
        raise FileNotFoundError(str(cf.path))
    taq = read_taq(cf)
    trades, quotes = split_trades_quotes(taq)
    if trades.height == 0 or len(quotes) == 0:
        return 0, 0
    bars_5s = build_5sec_bars_core(trades, quotes, root=slot, expiry=expiry, every="5s")
    if bars_5s.height == 0:
        return 0, 0
    bars = downsample_bars(bars_5s, target_every=horizon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(out_path, compression="zstd", compression_level=3)
    return trades.height, bars.height


def daterange(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--horizon", default="15m")
    p.add_argument("--out-root", required=True)
    p.add_argument("--algoseek-root", default=None)
    p.add_argument("--n-slots", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_root = Path(args.out_root)
    algoseek_root = Path(args.algoseek_root) if args.algoseek_root else None

    n_ok = n_skip = n_err = n_empty = n_no_data = 0
    errors: list[str] = []

    for d in daterange(start, end):
        contracts = front_n("VX", d, n=args.n_slots, dataset="vix",
                             algoseek_root=algoseek_root)
        if not contracts:
            n_no_data += 1
            continue
        for i, fc in enumerate(contracts, start=1):
            slot = f"VX{i}"
            out = _out_path(out_root, slot, args.horizon, d)
            if out.exists() and not args.overwrite:
                n_skip += 1
                continue
            try:
                n_trades, n_bars = build_slot_day(
                    slot, fc.expiry, d, args.horizon, out,
                    algoseek_root=algoseek_root,
                )
                if n_trades == 0:
                    n_empty += 1
                    print(f"[empty] {d}  {slot} {fc.expiry}  zero trades/bars")
                else:
                    n_ok += 1
                    print(f"[ok]    {d}  {slot} {fc.expiry}  trades={n_trades:,}  bars={n_bars}  → {out.name}")
            except Exception as e:
                n_err += 1
                msg = f"[err]  {d}  {slot} {fc.expiry}  {type(e).__name__}: {e}"
                print(msg, file=sys.stderr)
                errors.append(msg)
                traceback.print_exc(file=sys.stderr)

    print(f"\n[summary] VX [{start}..{end}] horizon={args.horizon} slots={args.n_slots}")
    print(f"  built={n_ok}  empty(holiday)={n_empty}  skipped(have)={n_skip}  no-data-day={n_no_data}  errors={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
