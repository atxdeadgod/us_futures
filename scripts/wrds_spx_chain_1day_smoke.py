"""Pull SPX options chain for 2024-01-02 locally to /tmp and verify parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2

OUT_DIR = Path("/tmp/spx_chain_smoke")
SPX_SECID = 108105


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[wrds] Connecting...", flush=True)
    try:
        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds",
            user="gargru", connect_timeout=30,
        )
        conn.set_session(readonly=True, autocommit=True)
    except Exception as e:
        print(f"[wrds] CONNECT FAILED: {e}", file=sys.stderr); return 1

    try:
        sql = """
            SELECT date, secid, symbol, strike_price, exdate, cp_flag,
                   best_bid, best_offer, volume, open_interest,
                   impl_volatility, delta, gamma, vega, theta,
                   optionid, expiry_indicator, root, suffix
            FROM optionm.opprcd2024
            WHERE secid = %s AND date = %s
            ORDER BY exdate, strike_price, cp_flag
        """
        df = pd.read_sql(sql, conn, params=(SPX_SECID, "2024-01-02"))
    finally:
        conn.close()

    print(f"\nrows: {len(df)}")
    if df.empty:
        print("EMPTY — something wrong")
        return 1
    # WRDS strike is stored × 1000
    df["strike_price"] = df["strike_price"] / 1000.0
    print("\nfirst 5 rows:")
    print(df.head())
    print("\nstrike range:", df["strike_price"].min(), "–", df["strike_price"].max())
    print("expiries:", sorted(df["exdate"].unique())[:10], "...", len(df["exdate"].unique()), "total")
    print("cp_flag counts:", df["cp_flag"].value_counts().to_dict())
    print("total OI sum:", int(df["open_interest"].sum()))
    nonzero_gamma = (df["gamma"].abs() > 1e-6).sum()
    print(f"options with non-zero gamma: {nonzero_gamma} / {len(df)}")

    out = OUT_DIR / "SPX_20240102.parquet"
    df.to_parquet(out, compression="zstd", compression_level=3, index=False)
    size = out.stat().st_size
    print(f"\nwrote {out} ({size/1024:.1f} KiB)")
    # Round-trip read check
    df2 = pd.read_parquet(out)
    assert df2.shape == df.shape
    print("parquet round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
