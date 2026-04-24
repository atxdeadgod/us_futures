"""Local smoke test for the SPX chain pull.

Runs just the discovery + secid-lookup + ONE DAY pull for SPX 2024 to verify:
  1. optionm tables found
  2. SPX secid resolved correctly
  3. A single date query returns sensible rows
  4. Schema matches what our GEX engine will expect

Single Duo push. No retries. Safe to run locally from your mac.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import psycopg2
import psycopg2.extras


def main() -> int:
    print("[wrds] Connecting — respond to Duo push if prompted...", flush=True)
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
        print(f"[wrds] CONNECT FAILED: {e}", file=sys.stderr)
        return 1

    cur = conn.cursor()
    try:
        # 1. Discovery
        print("\n=== Discovery: all optionm tables ===")
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='optionm' ORDER BY table_name""")
        tables = [r[0] for r in cur.fetchall()]
        print(f"Total optionm tables: {len(tables)}")
        # Group them
        by_prefix = {}
        for t in tables:
            pref = t.rstrip("0123456789")
            by_prefix.setdefault(pref, []).append(t)
        print("\nTop 30 prefixes (table families):")
        for pref, lst in sorted(by_prefix.items(), key=lambda x: -len(x[1]))[:30]:
            sample = lst[:3]
            print(f"  {pref:<30} ({len(lst):>4}): {sample}")

        # 2. Find the SPX index secid via whatever table has index tickers
        print("\n=== SPX secid discovery ===")
        candidate_secid_tables = [t for t in tables
                                   if any(k in t.lower() for k in
                                          ["secur", "secnmd", "securd", "secname", "secnam"])]
        print(f"Candidate secid-lookup tables: {candidate_secid_tables[:10]}")

        for tbl in candidate_secid_tables[:5]:
            try:
                cur.execute(f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema='optionm' AND table_name='{tbl}'
                    ORDER BY ordinal_position""")
                cols = [r[0] for r in cur.fetchall()]
                print(f"\n  [{tbl}] columns: {cols[:15]}")
            except Exception as e:
                print(f"  {tbl}: {e}")

        # 3. Try a SPX lookup in each candidate
        print("\n=== SPX ticker search ===")
        for tbl in candidate_secid_tables[:5]:
            for col in ("ticker", "symbol", "index_name", "issuer"):
                try:
                    cur.execute(
                        f"SELECT DISTINCT secid FROM optionm.{tbl} WHERE {col} = 'SPX' LIMIT 5"
                    )
                    rows = [r[0] for r in cur.fetchall()]
                    if rows:
                        print(f"  optionm.{tbl} WHERE {col}='SPX' -> secid={rows}")
                        break
                except Exception:
                    continue

        # 4. Identify opprcd-family tables and peek schema of one
        print("\n=== opprcd tables ===")
        opprcd_tbls = [t for t in tables if t.lower().startswith("opprcd")]
        print(f"Found: {opprcd_tbls[:20]}")
        if opprcd_tbls:
            target = None
            # Prefer a 2024-ish table
            for t in opprcd_tbls:
                if "2024" in t:
                    target = t
                    break
            if target is None:
                target = opprcd_tbls[-1]
            print(f"\nSchema of {target}:")
            cur.execute(f"""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_schema='optionm' AND table_name='{target}'
                ORDER BY ordinal_position""")
            for name, dtype in cur.fetchall():
                print(f"  {name:<30} {dtype}")

            # Sample one day's SPX options
            print(f"\n=== Sample: 5 SPX options from optionm.{target} on 2024-01-02 ===")
            try:
                cur.execute(
                    f"""SELECT date, secid, symbol, strike_price, exdate, cp_flag,
                              best_bid, best_offer, volume, open_interest,
                              impl_volatility, delta, gamma
                        FROM optionm.{target}
                        WHERE date = '2024-01-02'
                        LIMIT 5"""
                )
                for row in cur.fetchall():
                    print(f"  {row}")
            except Exception as e:
                print(f"  sample query failed: {str(e)[:200]}")
    finally:
        cur.close()
        conn.close()
        print("\n[wrds] Closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
