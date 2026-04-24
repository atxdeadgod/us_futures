"""Same comprehensive verification, using psycopg2 directly to dodge wrds+sqlalchemy bug."""
from __future__ import annotations

import sys
import psycopg2
import psycopg2.extras


def heading(title: str) -> None:
    print(f"\n{'='*70}\n=== {title}\n{'='*70}", flush=True)


def q(cur, sql: str):
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def print_rows(cols, rows, max_rows: int = 60, max_width: int = 200):
    if not rows:
        print("  (no rows)")
        return
    widths = [max(len(c), max(len(str(r[i])) for r in rows[:max_rows])) for i, c in enumerate(cols)]
    widths = [min(w, 50) for w in widths]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print("-" * min(sum(widths) + 2 * len(widths), max_width))
    for r in rows[:max_rows]:
        print(fmt.format(*[str(v)[:50] for v in r]))
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows)")


def main() -> int:
    print("[wrds] Connecting via psycopg2 — pgpass should authenticate without prompt...", flush=True)
    try:
        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="gargru",
            connect_timeout=30,
        )
        conn.set_session(readonly=True, autocommit=True)
    except Exception as e:
        print(f"[wrds] CONNECTION FAILED: {e}", file=sys.stderr)
        return 1

    cur = conn.cursor()
    try:
        # 1. CBOE schemas
        heading("CBOE schemas — table inventory")
        cols, rows = q(cur, "SELECT DISTINCT table_schema FROM information_schema.tables WHERE table_schema LIKE 'cboe%' ORDER BY table_schema")
        for (sch,) in rows:
            cols2, tabs = q(cur, f"SELECT table_name FROM information_schema.tables WHERE table_schema='{sch}' ORDER BY table_name")
            print(f"\n[{sch}] {len(tabs)} tables")
            print_rows(cols2, tabs, max_rows=80)

        # 2. tr_ds_fut detailed
        heading("tr_ds_fut — table list")
        cols, rows = q(cur, "SELECT table_name FROM information_schema.tables WHERE table_schema='tr_ds_fut' ORDER BY table_name")
        print_rows(cols, rows)

        for cand in ["dsfutcontrinfo", "wrds_fut_contract", "wrds_fut_series", "dsfutcontrval"]:
            heading(f"tr_ds_fut.{cand} — schema + sample")
            try:
                c1, r1 = q(cur, f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='tr_ds_fut' AND table_name='{cand}' ORDER BY ordinal_position")
                print_rows(c1, r1)
                c2, r2 = q(cur, f"SELECT * FROM tr_ds_fut.{cand} LIMIT 3")
                print("\nSample:")
                print_rows(c2, r2, max_rows=3, max_width=300)
            except Exception as exc:
                print(f"  err: {str(exc)[:200]}")
                conn.rollback() if not conn.autocommit else None

        # 3. optionm — vix/fut/spx/spy
        heading("optionm — VIX/futures/SPX-related tables")
        cols, rows = q(cur, "SELECT table_name FROM information_schema.tables WHERE table_schema='optionm' AND (LOWER(table_name) LIKE '%vix%' OR LOWER(table_name) LIKE '%fut%' OR LOWER(table_name) LIKE '%spx%' OR LOWER(table_name) LIKE '%spy%' OR LOWER(table_name) LIKE '%idx%') ORDER BY table_name")
        print_rows(cols, rows)

        # 4. cisdm
        heading("cisdm + cisdmsmp — derivatives research center")
        cols, rows = q(cur, "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema LIKE 'cisdm%' ORDER BY table_schema, table_name")
        print_rows(cols, rows)
        for s, t in rows:
            try:
                c1, r1 = q(cur, f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{s}' AND table_name='{t}' ORDER BY ordinal_position")
                print(f"\n  [{s}.{t}] cols: {', '.join(c[0] for c in r1)}")
            except Exception as exc:
                print(f"  err on {s}.{t}: {str(exc)[:150]}")

        # 5. wrdsapps event-study intraday
        heading("wrdsapps_evtstudy_intraday* — event-study intraday")
        cols, rows = q(cur, "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema LIKE 'wrdsapps_evtstudy_intraday%' ORDER BY table_schema, table_name")
        print_rows(cols, rows)
        for s, t in rows:
            c1, r1 = q(cur, f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{s}' AND table_name='{t}' ORDER BY ordinal_position")
            print(f"\n  [{s}.{t}] cols: {', '.join(c[0] for c in r1)}")

        # 6. Cross-schema: futures-metadata column names
        heading("Cross-schema: columns hinting at futures-contract metadata")
        cols, rows = q(cur, """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE LOWER(column_name) IN (
                'expiration','expiry','contract_month','open_interest','oi',
                'futures_symbol','contract_symbol','contract_code','contract_id',
                'futures_ticker','underlying_futures'
            )
            ORDER BY table_schema, table_name, column_name
        """)
        print(f"Total hits: {len(rows)}")
        print_rows(cols, rows, max_rows=80)

        # 7. Cross-schema: tables with intraday-time AND price columns
        heading("Cross-schema: tables with both intraday-time AND price columns (excluding equity TAQ)")
        cols, rows = q(cur, """
            WITH cset AS (
                SELECT table_schema, table_name, ARRAY_AGG(LOWER(column_name)) AS cols
                FROM information_schema.columns
                WHERE table_schema NOT LIKE 'taqm%'
                  AND table_schema NOT LIKE 'taqs%'
                  AND table_schema NOT LIKE 'issm%'
                  AND table_schema NOT LIKE 'pg_%'
                  AND table_schema != 'information_schema'
                GROUP BY table_schema, table_name
            )
            SELECT table_schema, table_name
            FROM cset
            WHERE (
                'time_m'=ANY(cols) OR 'time_micro'=ANY(cols) OR 'tstamp'=ANY(cols)
                OR 'time_ns'=ANY(cols) OR 'tradetime'=ANY(cols) OR 'timestamp'=ANY(cols)
                OR 'event_time'=ANY(cols) OR 'time'=ANY(cols) OR 'utctime'=ANY(cols)
            )
            AND ('price'=ANY(cols) OR 'last_trade'=ANY(cols) OR 'trade_price'=ANY(cols)
                 OR 'bid'=ANY(cols) OR 'ask'=ANY(cols) OR 'bidprice'=ANY(cols)
                 OR 'last'=ANY(cols))
            ORDER BY table_schema, table_name
        """)
        print(f"Hits: {len(rows)}")
        print_rows(cols, rows, max_rows=120)

    finally:
        cur.close()
        conn.close()
        print("\n[wrds] Connection closed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
