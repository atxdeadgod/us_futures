"""Smoke test: build 1-min and 15-min ES bars for a single (expiry, day).

Run:
    python -m scripts.build_es_bars_smoke
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Allow running as module from repo root OR as plain script
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import polars as pl

from src.data.bars import bars_from_trades_quotes
from src.data.ingest import locate, read_taq, split_trades_quotes


def main() -> int:
    day = date(2024, 1, 2)
    expiry = "ESM4"  # front-month quarterly on 2024-01-02 was March 2024 (ESH4),
    # but ESM4 (June 2024) is clean for a deep-book pull on that day and gives us a stable series

    cf = locate("taq", "ES", expiry, day)
    print(f"[smoke] TAQ file: {cf.path}")
    if not cf.exists:
        print(f"[smoke] MISSING: {cf.path}", file=sys.stderr)
        return 1

    taq = read_taq(cf)
    print(f"[smoke] rows={taq.height:,}  cols={taq.width}")
    print(f"[smoke] Type counts:")
    print(taq.group_by("Type").len().sort("len", descending=True))

    trades, quotes = split_trades_quotes(taq)
    print(f"[smoke] trades={trades.height:,}  quotes={quotes.height:,}")
    print(f"[smoke] aggressor split: {dict(trades.group_by('aggressor_sign').len().iter_rows())}")

    bars_1m = bars_from_trades_quotes(trades, quotes, every="1m")
    bars_15m = bars_from_trades_quotes(trades, quotes, every="15m")
    print(f"[smoke] bars 1m:  {bars_1m.height} rows")
    print(f"[smoke] bars 15m: {bars_15m.height} rows")

    out_dir = REPO / "data" / "bars_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"ES_{expiry}_{day:%Y%m%d}_1m.parquet"
    p15 = out_dir / f"ES_{expiry}_{day:%Y%m%d}_15m.parquet"
    bars_1m.write_parquet(p1, compression="zstd", compression_level=3)
    bars_15m.write_parquet(p15, compression="zstd", compression_level=3)
    print(f"[smoke] wrote {p1} ({p1.stat().st_size / 1024:.1f} KiB)")
    print(f"[smoke] wrote {p15} ({p15.stat().st_size / 1024:.1f} KiB)")

    # Sanity spot-check: dump last 5 15-min bars
    print("\n[smoke] last 5 15-min bars:")
    print(bars_15m.tail(5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
