"""Single-connection metadata scan of WRDS for futures intraday/TAQ datasets.

DISCIPLINE:
- ONE Duo push, ONE session
- NO retries on failure
- All queries executed within one connect/close cycle
- If connection fails, exits immediately with informative error

Run from local mac with valid ~/.pgpass.
"""
from __future__ import annotations

import sys

import wrds


def main() -> int:
    print("[wrds] Connecting — respond to Duo push if prompted...", flush=True)
    try:
        db = wrds.Connection(wrds_username="gargru", connect_args={"connect_timeout": 30})
    except Exception as e:
        print(f"[wrds] CONNECTION FAILED: {e}", file=sys.stderr)
        print("[wrds] Stopping. Do NOT retry — investigate first (stale pgpass? missed Duo?).", file=sys.stderr)
        return 1

    try:
        # 1. Find all schemas plausibly related to futures
        print("\n=== Schemas matching futures/CME/ICE/tick patterns ===", flush=True)
        schemas = db.raw_sql(
            """
            SELECT DISTINCT table_schema
            FROM information_schema.tables
            WHERE LOWER(table_schema) ~ '(fut|cme|ice|tick|nybot|nymex|cbot|comex|cbf|ftse)'
            ORDER BY table_schema
            """
        )
        print(schemas.to_string(index=False))

        # 2. List tables in each candidate schema
        for sch in schemas["table_schema"]:
            tabs = db.raw_sql(
                f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{sch}'
                ORDER BY table_name
                """
            )
            print(f"\n--- schema '{sch}' ({len(tabs)} tables) ---", flush=True)
            print(tabs.head(40).to_string(index=False))
            if len(tabs) > 40:
                print(f"... ({len(tabs) - 40} more)")

        # 3. Heuristically identify TAQ-like tables (look for time-resolved + price columns)
        print("\n=== TAQ-like candidate tables (have date+time+price columns) ===", flush=True)
        for sch in schemas["table_schema"]:
            cols = db.raw_sql(
                f"""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{sch}'
                  AND (LOWER(column_name) ~ '(time|tstamp|timestamp|usec|nsec|microsec)'
                       OR LOWER(column_name) IN ('price','last','trade_price','bid','ask','bidprice','askprice','quantity','size','volume'))
                ORDER BY table_name, column_name
                """
            )
            if len(cols):
                # group by table for compactness
                taq_tables = (
                    cols.groupby("table_name")["column_name"].apply(lambda x: ", ".join(sorted(set(x))))
                )
                # only show tables whose column set looks intraday-ish (has time and price/bid/ask)
                interesting = []
                for table_name, colstr in taq_tables.items():
                    has_time = any(k in colstr for k in ["time", "tstamp", "usec", "nsec", "microsec"])
                    has_px = any(k in colstr for k in ["price", "bid", "ask", "last"])
                    if has_time and has_px:
                        interesting.append((table_name, colstr))
                if interesting:
                    print(f"\n[{sch}]")
                    for tn, cs in interesting:
                        print(f"  {tn}: {cs}")

        # 4. Quick sample row count for any obviously-futures tables (single low-cost LIMIT 0 query)
        print("\n=== Confirming a couple of candidates with LIMIT 0 ===", flush=True)
        for candidate in ["tickdata.tdi_us_fut_taq", "cme.cmecont", "fut.fut", "tickdata"]:
            try:
                _ = db.raw_sql(f"SELECT * FROM {candidate} LIMIT 0")
                print(f"  {candidate}: OK")
            except Exception as exc:
                msg = str(exc).split("\n")[0][:100]
                print(f"  {candidate}: {msg}")

    finally:
        db.close()
        print("\n[wrds] Connection closed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
