"""Comprehensive single-session verification of WRDS futures coverage.

Multiple search angles in one connection:
  1. Drill cboe schemas (full table listing + sample columns)
  2. Drill tr_ds_fut (sample rows to verify contract codes/granularity)
  3. Drill optionm (look for futures-on-options or VIX)
  4. Drill cisdm (derivatives research center)
  5. Drill all wrdsapps_evtstudy_intraday* (event-study intraday)
  6. Search ALL schemas for any column named ticker/symbol/contract whose distinct
     values contain CME futures codes (ES*, NQ*, CL*, GC* etc.)
  7. Search ALL tables for column names suggesting tick-level futures
     (e.g., 'expiration', 'expiry', 'contract_month', 'open_interest')

ONE Duo push. ZERO retries.
"""
from __future__ import annotations

import sys
import time

import wrds


def heading(title: str) -> None:
    print(f"\n{'='*70}\n=== {title}\n{'='*70}", flush=True)


def main() -> int:
    print("[wrds] Connecting — respond to Duo push if prompted...", flush=True)
    try:
        db = wrds.Connection(wrds_username="gargru", connect_args={"connect_timeout": 30})
    except Exception as e:
        print(f"[wrds] CONNECTION FAILED: {e}", file=sys.stderr)
        return 1
    try:
        # 1. CBOE schemas
        heading("CBOE schemas — table inventory")
        cboe_schemas = db.raw_sql(
            "SELECT DISTINCT table_schema FROM information_schema.tables WHERE table_schema LIKE 'cboe%' ORDER BY table_schema"
        )
        for sch in cboe_schemas["table_schema"]:
            tabs = db.raw_sql(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema='{sch}' ORDER BY table_name"
            )
            print(f"\n[{sch}] {len(tabs)} tables")
            print(tabs.head(60).to_string(index=False))

        # 2. tr_ds_fut detailed
        heading("tr_ds_fut — table list + sample contracts")
        tabs = db.raw_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='tr_ds_fut' ORDER BY table_name"
        )
        print(tabs.to_string(index=False))
        # Try to get sample rows from a contract-info table
        for cand in ["dsfutcontrinfo", "wrds_fut_contract", "wrds_fut_series", "dsfutcontrval"]:
            try:
                cols = db.raw_sql(
                    f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='tr_ds_fut' AND table_name='{cand}' ORDER BY ordinal_position"
                )
                print(f"\n[tr_ds_fut.{cand}] columns:")
                print(cols.to_string(index=False))
                sample = db.raw_sql(f"SELECT * FROM tr_ds_fut.{cand} LIMIT 5")
                print(f"\n[tr_ds_fut.{cand}] sample rows:")
                print(sample.to_string(index=False)[:2000])
            except Exception as exc:
                print(f"  {cand}: {str(exc)[:120]}")

        # 3. optionm — look for VIX or futures-related tables
        heading("optionm — table list filtered for VIX/futures")
        tabs = db.raw_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='optionm' ORDER BY table_name"
        )
        print(f"Total optionm tables: {len(tabs)}")
        print("Sample (first 30):")
        print(tabs.head(30).to_string(index=False))
        vix_tabs = tabs[tabs["table_name"].str.contains("vix|fut|spx|spy", case=False, na=False)]
        print(f"\nFiltered to vix/fut/spx/spy ({len(vix_tabs)}):")
        print(vix_tabs.to_string(index=False))

        # 4. cisdm
        heading("cisdm — Center for International Securities and Derivatives")
        try:
            tabs = db.raw_sql(
                "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema LIKE 'cisdm%' ORDER BY table_schema, table_name"
            )
            print(tabs.to_string(index=False))
            for _, row in tabs.iterrows():
                cols = db.raw_sql(
                    f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{row['table_schema']}' AND table_name='{row['table_name']}' ORDER BY ordinal_position"
                )
                print(f"\n  [{row['table_schema']}.{row['table_name']}] columns: {', '.join(cols['column_name'])}")
        except Exception as exc:
            print(f"  cisdm error: {str(exc)[:120]}")

        # 5. wrdsapps event-study intraday
        heading("wrdsapps_evtstudy_intraday* — what's there")
        try:
            tabs = db.raw_sql(
                "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema LIKE 'wrdsapps_evtstudy_intraday%' ORDER BY table_schema, table_name"
            )
            print(tabs.to_string(index=False))
            for _, row in tabs.iterrows():
                cols = db.raw_sql(
                    f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{row['table_schema']}' AND table_name='{row['table_name']}'"
                )
                print(f"\n  [{row['table_schema']}.{row['table_name']}]: {', '.join(cols['column_name'])}")
        except Exception as exc:
            print(f"  wrdsapps error: {str(exc)[:120]}")

        # 6. Search ALL schemas for columns hinting at futures contract metadata
        heading("Cross-schema search: columns suggesting futures contracts")
        cols = db.raw_sql(
            """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE LOWER(column_name) IN (
                'expiration','expiry','contract_month','open_interest','oi',
                'futures_symbol','contract_symbol','contract_code','contract_id',
                'futures_ticker','underlying_futures'
            )
            ORDER BY table_schema, table_name, column_name
            """
        )
        print(f"Hits: {len(cols)}")
        if len(cols):
            print(cols.head(60).to_string(index=False))
            if len(cols) > 60:
                print(f"... ({len(cols) - 60} more)")

        # 7. Search ALL TAQ-like tables by column-set heuristic across ALL schemas
        heading("Cross-schema search: tables with both intraday-time AND price columns")
        rows = db.raw_sql(
            """
            WITH cset AS (
                SELECT table_schema, table_name, ARRAY_AGG(LOWER(column_name)) AS cols
                FROM information_schema.columns
                GROUP BY table_schema, table_name
            )
            SELECT table_schema, table_name, cols
            FROM cset
            WHERE (
                'time_m'=ANY(cols) OR 'time_micro'=ANY(cols) OR 'tstamp'=ANY(cols)
                OR 'time_ns'=ANY(cols) OR 'tradetime'=ANY(cols) OR 'timestamp'=ANY(cols)
                OR 'event_time'=ANY(cols)
            )
            AND ('price'=ANY(cols) OR 'last_trade'=ANY(cols) OR 'trade_price'=ANY(cols)
                 OR 'bid'=ANY(cols) OR 'ask'=ANY(cols) OR 'bidprice'=ANY(cols))
            ORDER BY table_schema, table_name
            """
        )
        print(f"Tables matching intraday-time + price columns: {len(rows)}")
        # Filter to non-equity-TAQ schemas to focus on futures candidates
        non_equity = rows[~rows["table_schema"].str.startswith(("taqm", "taqs", "issm"))]
        print(f"After excluding equity TAQ schemas: {len(non_equity)}")
        if len(non_equity):
            for _, r in non_equity.head(80).iterrows():
                print(f"  {r['table_schema']}.{r['table_name']}: {', '.join(sorted(r['cols'][:30]))[:200]}")

    finally:
        db.close()
        print("\n[wrds] Connection closed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
