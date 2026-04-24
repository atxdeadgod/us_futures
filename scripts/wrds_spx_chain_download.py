"""Pull daily SPX (+ SPXW + SPY) options chain from WRDS optionm for GEX computation.

DISCIPLINE (per our memory rules):
  - ONE Duo push per invocation
  - NO retries on connection failure
  - Single long-lived session for the whole pull

For each year in [start_year, end_year]:
  SELECT date, secid, ticker, strike_price, exdate, cp_flag, best_bid, best_offer,
         volume, open_interest, impl_volatility, delta, gamma, vega, theta
  FROM optionm.opprcd_<YEAR>
  WHERE ticker IN ('SPX', 'SPXW', 'SPY')
  AND date BETWEEN <start_date> AND <end_date>

Output: one parquet per (ticker, year) at:
  $SPX_CHAIN_ROOT/<ticker>/<YYYY>.parquet

Designed to run once via SLURM. Resumable: skips ticker-year pairs whose parquet
already exists with >1000 rows.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

DEFAULT_OUT_ROOT = Path(
    os.environ.get(
        "SPX_CHAIN_ROOT",
        "/N/project/ksb-finance-backtesting/data/spx_options_chain",
    )
)

TICKERS = ["SPX", "SPXW", "SPY"]

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
WRDS_USER = "gargru"


def get_conn():
    print(f"[wrds] Connecting — respond to Duo push if prompted...", flush=True)
    conn = psycopg2.connect(
        host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB, user=WRDS_USER,
        connect_timeout=30,
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


def secid_for_ticker(cur, ticker: str, year: int) -> list[int]:
    """Look up SPX/SPXW/SPY security IDs used by WRDS optionm for a given year."""
    cur.execute(
        """
        SELECT DISTINCT secid
        FROM optionm.secnmd
        WHERE ticker = %s
        """,
        (ticker,),
    )
    return [row[0] for row in cur.fetchall()]


def pull_year(cur, ticker: str, year: int, out_dir: Path) -> dict:
    out_file = out_dir / ticker / f"{year}.parquet"
    if out_file.exists() and out_file.stat().st_size > 10_000:
        try:
            rows_existing = pd.read_parquet(out_file).shape[0]
            if rows_existing > 1000:
                return {"status": "skipped-existing", "rows": rows_existing}
        except Exception:
            pass
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Secid lookup is per-year because optionm may reassign
    secids = secid_for_ticker(cur, ticker, year)
    if not secids:
        return {"status": f"no-secid for {ticker}/{year}", "rows": 0}

    # Some years reside in optionm.opprcd<yy>; others in opprcd_<yyyy>. Try both.
    # (Per WRDS docs the canonical is opprcd<yy> for legacy; opprcd_<yyyy> for newer.)
    candidate_tables = [f"optionm.opprcd{year % 100:02d}", f"optionm.opprcd_{year}"]
    rows = None
    last_err = None
    for table in candidate_tables:
        try:
            sql = f"""
                SELECT date, secid, symbol, strike_price, exdate, cp_flag,
                       best_bid, best_offer, volume, open_interest,
                       impl_volatility, delta, gamma, vega, theta
                FROM {table}
                WHERE secid IN %s
                ORDER BY date, secid, exdate, strike_price, cp_flag
            """
            cur.execute(sql, (tuple(secids),))
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
            source_table = table
            break
        except Exception as exc:
            last_err = str(exc)[:140]
            continue

    if rows is None:
        return {"status": f"failed-all-tables: {last_err}", "rows": 0}
    if not rows:
        return {"status": "empty-result", "rows": 0}

    df = pd.DataFrame(rows, columns=cols)
    # Normalize types
    if "strike_price" in df:
        df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce") / 1000.0  # WRDS optionm stores strike × 1000
    for col in ["best_bid", "best_offer", "impl_volatility", "delta", "gamma", "vega", "theta"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(out_file, compression="zstd", compression_level=3, index=False)
    return {
        "status": "ok",
        "rows": len(df),
        "bytes": out_file.stat().st_size,
        "source_table": source_table,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--tickers", default=",".join(TICKERS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    years = list(range(args.start_year, args.end_year + 1))

    try:
        conn = get_conn()
    except Exception as e:
        print(f"[wrds] CONNECTION FAILED: {e}", file=sys.stderr)
        print("[wrds] Do NOT retry — investigate (stale pgpass? missed Duo?).", file=sys.stderr)
        return 1
    cur = conn.cursor()

    print(f"[spx-chain] out_root={out_root}", flush=True)
    print(f"[spx-chain] tickers={tickers} years={years[0]}-{years[-1]}", flush=True)

    total_rows = 0
    total_bytes = 0
    try:
        for ticker in tickers:
            for year in years:
                result = pull_year(cur, ticker, year, out_root)
                total_rows += result.get("rows", 0)
                total_bytes += result.get("bytes", 0)
                print(f"[spx-chain] {ticker}/{year}: {result}", flush=True)
    finally:
        cur.close()
        conn.close()
        print(f"[spx-chain] DONE. total_rows={total_rows:,}  total_bytes={total_bytes/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
