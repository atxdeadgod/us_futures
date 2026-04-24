"""List ALL WRDS schemas accessible to gargru, with table counts.

Single connection, single Duo push, NO retries.
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
        print("[wrds] Stopping. Do NOT retry.", file=sys.stderr)
        return 1

    try:
        # Every schema the user can read from + table counts
        df = db.raw_sql(
            """
            SELECT table_schema AS schema, COUNT(*) AS n_tables
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog','information_schema')
            GROUP BY table_schema
            ORDER BY table_schema
            """
        )
        print(f"\n=== {len(df)} schemas accessible ===\n", flush=True)
        # Print all in a compact 3-column layout
        for i in range(0, len(df), 3):
            chunk = df.iloc[i : i + 3]
            line = "  ".join(f"{r['schema']:<32}({r['n_tables']:>4})" for _, r in chunk.iterrows())
            print(line)
    finally:
        db.close()
        print("\n[wrds] Connection closed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
