"""End-to-end verification on HPC: roll detection + ingest + bar build for ES on one day.

Run on HPC (bigred) after Phase 1 sync completes. Uses the live project-space data.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.bars import bars_from_trades_quotes
from src.data.ingest import locate, read_taq, split_trades_quotes
from src.data.roll import front_month, iter_front_series, detect_rolls


def main() -> int:
    # 1) Front-month detection for ES on several dates
    print("=== front-month on key dates ===")
    for d in [date(2024, 1, 2), date(2024, 3, 1), date(2024, 3, 15), date(2024, 6, 14)]:
        fc = front_month("ES", d, dataset="taq")
        print(f"  {d}: {fc}")

    # 2) Full series for Q1 2024 + roll events
    print("\n=== ES front series Q1 2024 ===")
    series = list(iter_front_series("ES", date(2024, 1, 2), date(2024, 3, 31)))
    print(f"  {len(series)} trading days")
    rolls = detect_rolls(series)
    print(f"  roll events: {rolls}")

    # 3) Ingest + bar build on one day (use first day from series)
    if not series:
        print("NO DATA — is this running on HPC or before sync completes?")
        return 1
    fc0 = series[0]
    print(f"\n=== ingest + bar build for {fc0.day} {fc0.expiry} ===")
    cf = locate("taq", "ES", fc0.expiry, fc0.day)
    print(f"  file: {cf.path}  ({fc0.bytes:,} bytes)")
    taq = read_taq(cf)
    print(f"  rows: {taq.height:,}")
    trades, quotes = split_trades_quotes(taq)
    print(f"  trades: {trades.height:,}  quotes: {quotes.height:,}")
    print(f"  aggressor counts: {dict(trades.group_by('aggressor_sign').len().iter_rows())}")
    print(f"  implied trades: {int(trades['is_implied'].sum())}")

    bars_1m = bars_from_trades_quotes(trades, quotes, every="1m")
    bars_15m = bars_from_trades_quotes(trades, quotes, every="15m")
    print(f"  bars 1m: {bars_1m.height} rows")
    print(f"  bars 15m: {bars_15m.height} rows")
    print("\n  sample 15m tail:")
    print(bars_15m.tail(5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
