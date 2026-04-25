"""Track C: build per-day GEX profile from SPX options chain.

Reads the SPX (and equivalents) options-chain parquets from the WRDS sync,
runs `compute_daily_gex_profile` for each ticker × year, writes the per-day
GEX profile parquet. The output (one row per trading date with columns
total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike, etc.) is the
input to `attach_gex_features` at feature-build time.

Inputs (per ticker, per year):
    {SPX_CHAIN_ROOT}/{TICKER}/{YEAR}.parquet   ← columns:
        date, secid, strike_price, exdate, cp_flag, open_interest, gamma

Spot prices: derived from the chain's at-the-money implied prices (or
external CRSP daily close, if available).  For V1 we use a simple
median-of-strikes proxy or pass via --spot-csv.

Outputs:
    {OUT_ROOT}/{TICKER}_gex_profile_{YEAR}.parquet
        date, total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike,
        max_put_oi_strike, gex_0dte_share, gex_0dte_only, gex_without_0dte

For the directional ES model, SPX is the primary driver.  V1 runs SPX only
(plus optionally SPY which tracks SPX 1:1).  Other tickers (NDX/QQQ/IWM/RUT/
DIA/DJX) are noted but not yet wired into the ES feature panel — they map
to NQ/RTY/YM respectively when those instruments get their own GEX features.

Usage:
    python scripts/build_gex_features.py \
        --tickers SPX,SPY \
        --start-year 2020 --end-year 2024 \
        --spx-chain-root /N/project/.../spx_options_chain \
        --spot-csv /N/project/.../spx_daily_close.csv \
        --out-root /N/project/.../gex_features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features.gex import compute_daily_gex_profile


def _build_spot_proxy_from_chain(chain: pl.DataFrame) -> pl.DataFrame:
    """Fallback spot estimator: per-date median strike weighted by total gamma·OI.

    This is a rough proxy used when no external spot CSV is provided. Better:
    pass --spot-csv with actual daily index closes from CRSP / Yahoo.
    """
    weighted = chain.with_columns(
        (pl.col("gamma") * pl.col("open_interest")).alias("_w")
    ).filter(pl.col("_w") > 0)
    spot = (
        weighted.group_by("date")
        .agg([
            (pl.col("strike_price") * pl.col("_w")).sum().alias("_num"),
            pl.col("_w").sum().alias("_den"),
        ])
        .with_columns((pl.col("_num") / pl.col("_den")).alias("spot"))
        .select(["date", "spot"])
    )
    return spot


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPX",
                   help="Comma-separated tickers; default SPX (V1 directional ES focus)")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--spx-chain-root", required=True,
                   help="Root containing {TICKER}/{YEAR}.parquet")
    p.add_argument("--spot-csv", default=None,
                   help="Optional CSV with columns [date, spot] per ticker (concatenated). "
                        "If absent, derive spot proxy from chain.")
    p.add_argument("--out-root", required=True)
    args = p.parse_args()

    chain_root = Path(args.spx_chain_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    spot_csv = None
    if args.spot_csv:
        spot_csv = pl.read_csv(args.spot_csv).with_columns(pl.col("date").cast(pl.Date))

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    n_ok = n_err = 0
    for ticker in tickers:
        for yr in range(args.start_year, args.end_year + 1):
            in_path = chain_root / ticker / f"{yr}.parquet"
            if not in_path.exists():
                print(f"[skip] {ticker} {yr}: chain parquet missing → {in_path}", file=sys.stderr)
                continue
            try:
                chain = pl.read_parquet(in_path).with_columns([
                    pl.col("date").cast(pl.Date),
                    pl.col("exdate").cast(pl.Date),
                    pl.col("cp_flag").cast(pl.Utf8),
                    pl.col("strike_price").cast(pl.Float64),
                    pl.col("open_interest").cast(pl.Float64),
                    pl.col("gamma").cast(pl.Float64),
                ])
                if spot_csv is not None and "ticker" in spot_csv.columns:
                    spot = spot_csv.filter(pl.col("ticker") == ticker).select(["date", "spot"])
                else:
                    spot = _build_spot_proxy_from_chain(chain)

                profile = compute_daily_gex_profile(chain, spot)
                out_path = out_root / f"{ticker}_gex_profile_{yr}.parquet"
                profile.write_parquet(out_path, compression="zstd", compression_level=3)
                print(f"[ok]   {ticker} {yr}: {profile.height} dates → {out_path.name}")
                n_ok += 1
            except Exception as e:
                print(f"[err]  {ticker} {yr}: {type(e).__name__}: {e}", file=sys.stderr)
                n_err += 1

    print(f"\n[summary] built={n_ok} errors={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
