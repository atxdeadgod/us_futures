"""Build the FUTURES feature panel for one (instrument, year).

Reads:
    - Phase A+B bars (or Phase A if --no-l2-deep) for the target
    - VX1/VX2/VX3 Phase A bars (if not --no-vx)

Produces:
    {OUT_ROOT}/futures/{INSTR}_{YEAR}.parquet

Pipeline (all bar-derived; no options or labels here):
    bars → build_per_instrument_features
         → attach_l2_deep_features (Phase A+B only)
         → attach_vx_features (asof-join VX1/2/3)

The output is the "single-source futures panel". `build_options_panel.py` is
the parallel data-source pass; `build_single_panel.py` joins them and labels.

Usage:
    python scripts/build_futures_panel.py --instrument ES --year 2024 \
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

from src.features import external_sources, single_contract, sub_bar_engines


def _stitch_per_day_parquets(root: Path, instr: str, horizon: str,
                             year: int) -> pl.DataFrame:
    folder = root / instr / horizon
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = sorted(folder.glob(f"{instr}_{year}*_{horizon}.parquet"))
    if not files:
        raise FileNotFoundError(f"No {instr} parquets in {folder} for year {year}")
    # diagonal_relaxed: aligns columns by name, fills missing with nulls, AND
    # relaxes dtype mismatches. Necessary because some years' bars have a
    # narrower schema than others (e.g., 2020-Apr2023 Phase A+B lacks
    # bid_sz_L10/ask_sz_L10 due to the Algoseek "Leve101Size" header typo).
    # Plain vertical_relaxed only relaxes dtypes, not column sets, and would
    # crash with "schema lengths differ".
    return pl.concat([pl.read_parquet(p) for p in files], how="diagonal_relaxed").sort("ts")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True, help="Output root; futures panel writes to {out}/futures/")
    p.add_argument("--bars-phase-a-root", default="/N/project/ksb-finance-backtesting/data/bars_phase_a")
    p.add_argument("--bars-phase-ab-root", default="/N/project/ksb-finance-backtesting/data/bars_phase_ab")
    p.add_argument("--bars-5s-root", default="/N/project/ksb-finance-backtesting/data/bars_5sec",
                   help="Root for 5-sec bars (used for VPIN + Hawkes engines). Skip if not present.")
    p.add_argument("--bars-phase-e-root", default="/N/project/ksb-finance-backtesting/data/bars_phase_e",
                   help="Root for Phase E bars (eff spread, large-trade, hidden, cancel, quote count).")
    p.add_argument("--horizon", default="15m")
    p.add_argument("--no-l2-deep", action="store_true",
                   help="Skip L2-deep features (use Phase A only)")
    p.add_argument("--no-vx", action="store_true")
    p.add_argument("--no-sub-bar-engines", action="store_true",
                   help="Skip VPIN + Hawkes (sub-bar engine) features")
    p.add_argument("--no-phase-e", action="store_true",
                   help="Skip Phase E (execution-quality + cancel-proxy + quote-event) features")
    p.add_argument("--vpin-bucket-size", type=int, default=25_000)
    p.add_argument("--hawkes-hl-fast", type=float, default=5.0)
    p.add_argument("--hawkes-hl-slow", type=float, default=60.0)
    args = p.parse_args()

    out_root = Path(args.out) / "futures"
    out_root.mkdir(parents=True, exist_ok=True)
    bars_phase_a = Path(args.bars_phase_a_root)
    bars_phase_ab = Path(args.bars_phase_ab_root)

    # Auto-detect: fall back to Phase A only if Phase A+B isn't built for this
    # instrument. The 26 macro/macro-feed contracts (GC, CL, 6E, ZN, ...) don't
    # have Phase A+B (depth bars) — only Phase A. Without this auto-fallback,
    # the multi-instrument production run would crash on every non-trading-4
    # contract.
    if not args.no_l2_deep:
        if not (bars_phase_ab / args.instrument / args.horizon).exists():
            print(f"[auto] Phase A+B not found for {args.instrument} at {bars_phase_ab}; "
                  f"falling back to Phase A only (skipping L2-deep features)")
            args.no_l2_deep = True
    src_root = bars_phase_a if args.no_l2_deep else bars_phase_ab
    print(f"[load] {args.instrument} {args.horizon} from {src_root}")
    bars = _stitch_per_day_parquets(src_root, args.instrument, args.horizon, args.year)
    print(f"  loaded {bars.height:,} bars  cols={len(bars.columns)}")

    print("[features] per-instrument core pass")
    feat = single_contract.build_per_instrument_features(
        bars,
        lookback_days_grid=(30, 60),
        attach_overnight=True,
        attach_patterns=True,
        attach_engines=True,
        attach_smoothed=True,
        attach_cyclic_minute=True,
    )
    print(f"  after core pass: cols={len(feat.columns)}")

    if not args.no_l2_deep:
        print("[features] L2 deep + per-level")
        feat = single_contract.attach_l2_deep_features(feat, depth=10, spread_z_window=60)
        print(f"  after L2 deep: cols={len(feat.columns)}")

    if not args.no_vx:
        try:
            print("[features] VX1/VX2/VX3 asof-join")
            vx1 = _stitch_per_day_parquets(bars_phase_a, "VX1", args.horizon, args.year)
            vx2 = _stitch_per_day_parquets(bars_phase_a, "VX2", args.horizon, args.year)
            vx3 = _stitch_per_day_parquets(bars_phase_a, "VX3", args.horizon, args.year)
            feat = external_sources.attach_vx_features(
                feat, vx1_bars=vx1, vx2_bars=vx2, vx3_bars=vx3,
            )
            print(f"  after VX: cols={len(feat.columns)}")
        except FileNotFoundError as e:
            print(f"  [skip-vx] {e}")

    if not args.no_sub_bar_engines:
        bars_5s_root = Path(args.bars_5s_root)
        try:
            print(f"[features] sub-bar engines (VPIN + Hawkes) from {bars_5s_root}")
            bars_5s = _stitch_per_day_parquets(bars_5s_root, args.instrument, "5s", args.year)
            print(f"  loaded {bars_5s.height:,} 5-sec bars")
            feat = sub_bar_engines.attach_sub_bar_engine_features(
                feat, bars_5s,
                vpin_bucket_size=args.vpin_bucket_size,
                hawkes_hl_fast=args.hawkes_hl_fast,
                hawkes_hl_slow=args.hawkes_hl_slow,
            )
            print(f"  after sub-bar engines: cols={len(feat.columns)}")
        except FileNotFoundError as e:
            print(f"  [skip-sub-bar] {e}")

    if not args.no_phase_e:
        bars_phase_e_root = Path(args.bars_phase_e_root)
        try:
            print(f"[features] Phase E (eff spread + large-trade + hidden + cancel + quote count) from {bars_phase_e_root}")
            phase_e_bars = _stitch_per_day_parquets(bars_phase_e_root, args.instrument, args.horizon, args.year)
            print(f"  loaded {phase_e_bars.height:,} Phase E bars  cols={len(phase_e_bars.columns)}")
            feat = single_contract.attach_phase_e_features(feat, phase_e_bars)
            print(f"  after Phase E: cols={len(feat.columns)}")
        except FileNotFoundError as e:
            print(f"  [skip-phase-e] {e}")

    out_path = out_root / f"{args.instrument}_{args.year}.parquet"
    feat.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[done] {out_path}  rows={feat.height:,}  cols={len(feat.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
