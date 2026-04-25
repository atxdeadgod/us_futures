"""Build per-trading-instrument feature panel by orchestrating panel.py steps.

Workflow:
    1. Load Phase A bars for all 30 contracts (universe) on the IS window
    2. For each contract: compute base microstructure features + TC/MAD z-scores
    3. Wide-join all 30 on ts → cross-asset frame
    4. Add cross-sectional Gauss-Rank features (universe + within asset class)
    5. Add cross-asset composites (synthetic DXY, rates curve, rolling corrs)
    6. Per target trading instrument: assemble panel, attach labels, write parquet

Output:
    {OUT_ROOT}/{TARGET}_features_panel_{START_YEAR}_{END_YEAR}.parquet

The 2024+ window is OUT-OF-SAMPLE — caller must NOT pass --end >= 2024-01-01
unless explicitly building the held-out OOS panel.

Usage:
    python scripts/build_features_panel.py \
        --target ES \
        --bars-phase-a-root /N/.../bars_phase_a \
        --start 2020-01-01 --end 2023-12-31 \
        --out-root /N/.../feature_panels
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import date
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import panel
from src.features.tc_features import attach_session_flags


def _load_phase_a_bars_for_instrument(
    bars_phase_a_root: Path, instr: str, start: date, end: date,
) -> pl.DataFrame | None:
    """Load all 15-min Phase A parquets for one instrument across [start, end]."""
    paths = sorted(glob.glob(
        str(bars_phase_a_root / instr / "15m" / f"{instr}_*_15m.parquet")
    ))
    if not paths:
        return None
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical_relaxed")
    df = lf.filter(
        (pl.col("ts").dt.date() >= start) & (pl.col("ts").dt.date() <= end)
    ).sort("ts").collect()
    return df if df.height > 0 else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=panel.TRADING_INSTRUMENTS)
    p.add_argument("--bars-phase-a-root", required=True,
                   help="Root containing {INSTR}/15m/{INSTR}_{YYYYMMDD}_15m.parquet")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--out-root", required=True)
    p.add_argument("--lookback-days-grid", default="30,60",
                   help="Comma-separated TS-normalization lookbacks")
    p.add_argument("--rolling-corr-window", type=int, default=60,
                   help="Window for cross-asset rolling correlations")
    p.add_argument("--allow-oos", action="store_true",
                   help="Override the 2024-01-01 OOS guardrail (use only for explicit OOS panels)")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end >= date(2024, 1, 1) and not args.allow_oos:
        raise SystemExit(
            f"ERROR: --end={end} hits OOS window (>= 2024-01-01). "
            "Pass --allow-oos to override; otherwise use 2023-12-31 or earlier."
        )

    bars_phase_a_root = Path(args.bars_phase_a_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lookback_grid = tuple(int(x) for x in args.lookback_days_grid.split(",") if x.strip())

    # ---- Step 1+2: load bars and compute per-instrument features ----
    print(f"[panel] target={args.target}  IS=[{start}..{end}]  lookback_grid={lookback_grid}")
    per_instr: dict[str, pl.DataFrame] = {}
    for instr in panel.ALL_INSTRUMENTS:
        df = _load_phase_a_bars_for_instrument(bars_phase_a_root, instr, start, end)
        if df is None:
            print(f"  [skip] {instr}: no Phase A bars in range", file=sys.stderr)
            continue
        df = attach_session_flags(df)
        df = panel.build_per_instrument_features(df, lookback_days_grid=lookback_grid)
        per_instr[instr] = df
        print(f"  [ok]   {instr}: {df.height:,} bars")

    if args.target not in per_instr:
        raise SystemExit(f"target={args.target} has no Phase A bars in range; aborting")

    # ---- Step 3: wide cross-asset join ----
    wide = panel.build_wide_cross_asset_frame(per_instr, base_value_cols=panel.BASE_VALUE_COLS)
    print(f"[panel] wide frame: {wide.height:,} ts x {wide.width} cols")

    # ---- Step 4: cross-sectional Gauss-Rank ----
    wide = panel.attach_cross_sectional_ranks(
        wide, base_value_cols=panel.BASE_VALUE_COLS,
        instruments=panel.ALL_INSTRUMENTS, asset_classes=panel.ASSET_CLASSES,
    )
    print(f"[panel] + cross-sectional ranks: {wide.width} cols")

    # ---- Step 5: cross-asset composites ----
    wide = panel.attach_cross_asset_composites(wide, rolling_corr_window=args.rolling_corr_window)
    print(f"[panel] + composites: {wide.width} cols")

    # ---- Step 6: assemble target panel + labels ----
    out_panel = panel.assemble_target_panel(
        target=args.target,
        target_bars_with_features=per_instr[args.target],
        wide_cross_asset=wide,
    )
    print(f"[panel] target panel: {out_panel.height:,} valid bars x {out_panel.width} cols")

    # ---- Write ----
    out_path = out_root / f"{args.target}_features_panel_{start.year}_{end.year}.parquet"
    out_panel.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[panel] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
