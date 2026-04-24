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


def discover_tables(cur) -> dict:
    """Discover actual table names in optionm — don't guess, ask."""
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'optionm'
        ORDER BY table_name
        """
    )
    all_tables = [r[0] for r in cur.fetchall()]
    # Options pricing tables (by year and by view)
    opprcd_by_year = {}
    opprcd_unified = None
    secid_tables = []
    for t in all_tables:
        low = t.lower()
        # Match opprcd1996, opprcd2014, opprcd_2024 etc.
        if low.startswith("opprcd"):
            rest = low[len("opprcd"):]
            if rest.startswith("_"):
                rest = rest[1:]
            if rest.isdigit() and len(rest) == 4:
                opprcd_by_year[int(rest)] = t
            elif rest.isdigit() and len(rest) == 2:
                # 2-digit year → map to 4-digit
                y2 = int(rest)
                yr = 1900 + y2 if y2 >= 50 else 2000 + y2
                opprcd_by_year[yr] = t
            elif rest == "":
                opprcd_unified = t
        if low in ("securd1", "secnmd", "securities", "secur") or low.startswith("secname") or low.startswith("security"):
            secid_tables.append(t)
    print(f"[discover] optionm tables total: {len(all_tables)}", flush=True)
    print(f"[discover] opprcd by year: {sorted(opprcd_by_year.items())[-10:] if opprcd_by_year else '(none)'}", flush=True)
    print(f"[discover] opprcd unified: {opprcd_unified!r}", flush=True)
    print(f"[discover] secid tables: {secid_tables}", flush=True)
    return {
        "opprcd_by_year": opprcd_by_year,
        "opprcd_unified": opprcd_unified,
        "secid_tables": secid_tables,
        "all_tables": all_tables,
    }


def secid_for_ticker(cur, ticker: str, secid_tables: list[str]) -> list[int]:
    """Look up SPX/SPXW/SPY security IDs; tries each candidate secid table."""
    for tbl in secid_tables:
        for col_ticker in ("ticker", "symbol", "issuer"):
            try:
                cur.execute(
                    f"SELECT DISTINCT secid FROM optionm.{tbl} WHERE {col_ticker} = %s",
                    (ticker,),
                )
                rows = [r[0] for r in cur.fetchall()]
                if rows:
                    return rows
            except Exception:
                continue
    # Last resort: cols may be called index_flag + index_name
    return []


def pull_year(cur, ticker: str, year: int, out_dir: Path, discovered: dict, secids: list[int]) -> dict:
    out_file = out_dir / ticker / f"{year}.parquet"
    if out_file.exists() and out_file.stat().st_size > 10_000:
        try:
            rows_existing = pd.read_parquet(out_file).shape[0]
            if rows_existing > 1000:
                return {"status": "skipped-existing", "rows": rows_existing}
        except Exception:
            pass
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not secids:
        return {"status": f"no-secid for {ticker}", "rows": 0}

    # Candidate sources: per-year table, then unified-with-date-filter
    candidates = []
    if year in discovered["opprcd_by_year"]:
        candidates.append(("optionm." + discovered["opprcd_by_year"][year], None))
    if discovered["opprcd_unified"]:
        # Use unified with date filter
        date_lo = f"{year}-01-01"
        date_hi = f"{year}-12-31"
        candidates.append(("optionm." + discovered["opprcd_unified"], (date_lo, date_hi)))

    if not candidates:
        return {"status": f"no-table-for-year-{year}", "rows": 0}

    rows = None
    cols = None
    source_table = None
    last_err = None
    for table, date_range in candidates:
        try:
            if date_range is None:
                sql = f"""
                    SELECT date, secid, strike_price, exdate, cp_flag,
                           best_bid, best_offer, volume, open_interest,
                           impl_volatility, delta, gamma
                    FROM {table}
                    WHERE secid IN %s
                    ORDER BY date, secid, exdate, strike_price, cp_flag
                """
                cur.execute(sql, (tuple(secids),))
            else:
                sql = f"""
                    SELECT date, secid, symbol, strike_price, exdate, cp_flag,
                           best_bid, best_offer, volume, open_interest,
                           impl_volatility, delta, gamma, vega, theta
                    FROM {table}
                    WHERE secid IN %s AND date >= %s AND date <= %s
                    ORDER BY date, secid, exdate, strike_price, cp_flag
                """
                cur.execute(sql, (tuple(secids), date_range[0], date_range[1]))
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
            source_table = table
            break
        except Exception as exc:
            last_err = str(exc)[:200]
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
        discovered = discover_tables(cur)
        # Secid lookup per ticker (same across years)
        ticker_secids = {}
        for ticker in tickers:
            sids = secid_for_ticker(cur, ticker, discovered["secid_tables"])
            ticker_secids[ticker] = sids
            print(f"[spx-chain] secids for {ticker}: {sids}", flush=True)

        for ticker in tickers:
            for year in years:
                result = pull_year(cur, ticker, year, out_root, discovered, ticker_secids[ticker])
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
