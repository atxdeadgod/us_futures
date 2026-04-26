"""Build the CROSS-SECTIONAL feature panel for one (target, year).

Reads all 30 per-instrument single panels for the year and produces a
target-specific cross-sectional feature panel:

    {OUT_ROOT}/cross/{TARGET}_{YEAR}.parquet

Structure of the output: target's full single panel (378+ cols) plus
cross-sectional columns:
    - cs_universe_*  Gauss-Rank within the 30-contract universe
    - cs_class_*     Gauss-Rank within asset class
    - synthetic_dxy_logret + rates curve spreads
    - corr_{target}_vs_{ref}_w60  cross-asset rolling correlations

Pipeline:
    load all 30 single panels, keep ts + CS_VALUE_COLS, prefix per instrument
    → build_wide_cross_asset_frame
    → attach_cross_sectional_ranks (universe + per-class)
    → attach_cross_asset_composites (DXY, rates curve, rolling corrs)
    → drop {target}_* cols + corrs not for this target
    → left-join onto target's single panel
    → write

Usage:
    python scripts/build_cross_panel.py --target ES --year 2024 \
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

from src.features import cross_sectional

# Subset of BASE_VALUE_COLS used as cross-sectional inputs. Two tiers:
# Tier 1 (directional): "is this instrument outperforming peers?"
# Tier 2 (vol/regime):   "is this instrument's vol regime unusual vs peers?"
# Pure-noise / sparse / contract-scale features (implied volume, large_trade,
# spread_to_mid_bps absolute) intentionally excluded — they don't normalize
# cleanly across the 30-contract universe.
CS_VALUE_COLS: list[str] = [
    # Tier 1 — directional
    "log_return", "abs_log_return", "log_volume",
    "ofi", "aggressor_ratio", "cvd_change",
    "vwap_deviation",
    # Tier 2 — vol / regime
    "realized_vol_w20", "realized_vol_w60", "realized_vol_w120",
    "vol_surprise_w20", "vol_surprise_w60",
    "amihud_illiq_w20", "range_vol_parkinson_w20",
]


def _load_single_panel(out_root: Path, instr: str, year: int) -> pl.DataFrame | None:
    path = out_root / "single" / f"{instr}_{year}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path).sort("ts")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=cross_sectional.TRADING_INSTRUMENTS)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rolling-corr-window", type=int, default=60)
    args = p.parse_args()

    out_root = Path(args.out)
    cross_dir = out_root / "cross"
    cross_dir.mkdir(parents=True, exist_ok=True)

    target_panel = _load_single_panel(out_root, args.target, args.year)
    if target_panel is None:
        print(f"[error] target single panel missing: {args.target}_{args.year}")
        return 1
    print(f"[load] target {args.target} single panel: rows={target_panel.height:,}  cols={len(target_panel.columns)}")

    per_instrument: dict[str, pl.DataFrame] = {}
    missing: list[str] = []
    for instr in cross_sectional.ALL_INSTRUMENTS:
        df = _load_single_panel(out_root, instr, args.year)
        if df is None:
            missing.append(instr)
            continue
        avail = [c for c in CS_VALUE_COLS if c in df.columns]
        if not avail:
            missing.append(instr)
            continue
        per_instrument[instr] = df.select(["ts"] + avail)
    if missing:
        print(f"[warn] missing single panels for: {missing}")
    print(f"[wide] joining {len(per_instrument)} instrument frames on ts")

    wide = cross_sectional.build_wide_cross_asset_frame(
        per_instrument, base_value_cols=CS_VALUE_COLS,
    )
    print(f"  wide frame: rows={wide.height:,}  cols={len(wide.columns)}")

    print("[cs] gauss-rank universe + per-class")
    wide = cross_sectional.attach_cross_sectional_ranks(
        wide,
        base_value_cols=CS_VALUE_COLS,
        instruments=list(per_instrument.keys()),
        asset_classes=cross_sectional.ASSET_CLASSES,
    )
    print(f"  after CS ranks: cols={len(wide.columns)}")

    print("[cs] composites (DXY, rates curve, rolling correlations)")
    wide = cross_sectional.attach_cross_asset_composites(
        wide, rolling_corr_window=args.rolling_corr_window,
    )
    print(f"  after composites: cols={len(wide.columns)}")

    # Filter wide frame: drop target's per-instrument prefix cols (already in
    # target's single panel) and drop other targets' corr cols
    target_prefix = f"{args.target}_"
    other_targets = [t for t in cross_sectional.TRADING_INSTRUMENTS if t != args.target]
    keep = ["ts"]
    for c in wide.columns:
        if c == "ts":
            continue
        if c.startswith(target_prefix):
            continue  # target's own values are in target_panel
        if any(c.startswith(f"corr_{t}_") for t in other_targets):
            continue  # corrs for other targets
        keep.append(c)
    wide_filtered = wide.select(keep)
    print(f"  filtered to target {args.target}: cols={len(wide_filtered.columns)}")

    # Left-join onto target's single panel
    already = set(target_panel.columns)
    add_cols = ["ts"] + [c for c in wide_filtered.columns if c != "ts" and c not in already]
    out = target_panel.join(wide_filtered.select(add_cols), on="ts", how="left")
    print(f"[merge] target + cross: rows={out.height:,}  cols={len(out.columns)}")

    # Regime × direction interactions (must run AFTER the merge so the
    # interaction function can see both target's per-bar features AND the
    # cross-asset composites that came from the wide frame).
    print("[cs] regime × direction interactions")
    out = cross_sectional.attach_regime_interactions(
        out, target=args.target, rolling_corr_window=args.rolling_corr_window,
    )
    print(f"  after interactions: cols={len(out.columns)}")

    out_path = cross_dir / f"{args.target}_{args.year}.parquet"
    out.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[done] {out_path}  rows={out.height:,}  cols={len(out.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
