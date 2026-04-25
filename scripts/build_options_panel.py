"""Build the OPTIONS-derived feature panel for one (target_instrument, year).

For V1 the only options-source feature group is GEX (gamma exposure profile,
SPX → ES; later NDX → NQ).

Reads:
    - GEX daily profile parquet (built by scripts/build_gex_features.py)
    - Bars for the target (just ts + close, used by attach_gex_for_target)

Produces:
    {OUT_ROOT}/options/{TARGET}_{YEAR}.parquet
    Schema: ts + GEX columns (no bar identity / no labels).

For non-equity-index targets there's no options-chain mapping yet; this script
exits cleanly with a "no chain configured" message.

Usage:
    python scripts/build_options_panel.py --instrument ES --year 2024 \
        --out /N/.../features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import external_sources

# Map: futures target → options ticker whose chain feeds it
TARGET_TO_OPTIONS_TICKER = {
    "ES": "SPX",
    # "NQ": "NDX",  # V1.5
}


def _stitch_per_day_parquets(root: Path, instr: str, horizon: str,
                             year: int) -> pl.DataFrame:
    folder = root / instr / horizon
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = sorted(folder.glob(f"{instr}_{year}*_{horizon}.parquet"))
    if not files:
        raise FileNotFoundError(f"No {instr} parquets in {folder} for year {year}")
    return pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed").sort("ts")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True, help="Output root; options panel writes to {out}/options/")
    p.add_argument("--bars-root", default="/N/project/ksb-finance-backtesting/data/bars_phase_ab",
                   help="Where to read target's bars (just for ts + close)")
    p.add_argument("--gex-root", default="/N/project/ksb-finance-backtesting/data/gex_features")
    p.add_argument("--horizon", default="15m")
    args = p.parse_args()

    if args.instrument not in TARGET_TO_OPTIONS_TICKER:
        print(f"[skip] no options chain configured for {args.instrument!r}; nothing to do.")
        return 0

    options_ticker = TARGET_TO_OPTIONS_TICKER[args.instrument]
    out_root = Path(args.out) / "options"
    out_root.mkdir(parents=True, exist_ok=True)

    gex_path = Path(args.gex_root) / f"{options_ticker}_gex_profile_{args.year}.parquet"
    if not gex_path.exists():
        print(f"[error] GEX profile missing: {gex_path}")
        return 1
    print(f"[load] GEX profile: {gex_path}")

    bars_root = Path(args.bars_root)
    print(f"[load] {args.instrument} {args.horizon} bars (ts+close) from {bars_root}")
    bars = _stitch_per_day_parquets(bars_root, args.instrument, args.horizon, args.year)
    bars_min = bars.select(["ts", "close"])
    print(f"  loaded {bars_min.height:,} bars")

    feat = external_sources.attach_gex_for_target(bars_min, [str(gex_path)])
    # Drop close — options panel keeps just ts + GEX columns; close lives in futures panel
    keep = ["ts"] + [c for c in feat.columns if c not in ("ts", "close")]
    feat = feat.select(keep)
    print(f"[features] GEX attached: cols={len(feat.columns)}")

    out_path = out_root / f"{args.instrument}_{args.year}.parquet"
    feat.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[done] {out_path}  rows={feat.height:,}  cols={len(feat.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
