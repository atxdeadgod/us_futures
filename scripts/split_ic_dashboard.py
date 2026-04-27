"""Split per-target IC dashboard CSV into single-asset and cross-asset CSVs.

Cross-asset classification rules (any one matches → cross-asset):
  - prefix `cs_universe_` or `cs_class_`         (Gauss-Rank cross-sectional)
  - prefix `ix_`                                 (regime × direction interactions)
  - prefix `corr_`                               (cross-asset rolling correlations)
  - exact `synthetic_dxy_logret`, `risk_off_score`, `butterfly_2s5s10s`
  - prefix `slope_2s5s_`, `slope_5s10s_`, `slope_2s10s_`, `slope_10s30s_`
  - starts with any non-target instrument prefix from the 30-contract universe

Everything else is single-asset (target's own features).

Usage:
    python scripts/split_ic_dashboard.py [--in-dir DIR] [--targets ES NQ RTY YM]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ASSET_CLASSES: dict[str, list[str]] = {
    "EQUITY_INDEX": ["ES", "NQ", "RTY", "YM"],
    "FX":           ["6A", "6B", "6C", "6E", "6J"],
    "ENERGY":       ["BZ", "CL", "HO", "NG", "RB"],
    "METALS":       ["GC", "HG", "PA", "PL", "SI"],
    "RATES":        ["SR3", "TN", "ZB", "ZF", "ZN", "ZT"],
    "AGS":          ["ZC", "ZL", "ZM", "ZS", "ZW"],
}
ALL_INSTRUMENTS = [c for cls in ASSET_CLASSES.values() for c in cls]

CROSS_PREFIXES_GENERIC = (
    "cs_universe_",
    "cs_class_",
    "ix_",
    "corr_",
    "slope_2s5s_",
    "slope_5s10s_",
    "slope_2s10s_",
    "slope_10s30s_",
)
CROSS_EXACT = {"synthetic_dxy_logret", "risk_off_score", "butterfly_2s5s10s"}


def is_cross_asset(feature: str, target: str) -> bool:
    if feature in CROSS_EXACT:
        return True
    if feature.startswith(CROSS_PREFIXES_GENERIC):
        return True
    # Other-instrument prefixes (target's own bare features stay single-asset)
    for instr in ALL_INSTRUMENTS:
        if instr == target:
            continue
        if feature.startswith(f"{instr}_"):
            return True
    return False


def split_one(in_path: Path, out_dir: Path, target: str) -> tuple[int, int]:
    df = pl.read_csv(in_path)
    df = df.with_columns(
        pl.col("feature").map_elements(
            lambda f: is_cross_asset(f, target), return_dtype=pl.Boolean
        ).alias("_is_cross")
    )
    cross = df.filter(pl.col("_is_cross")).drop("_is_cross")
    single = df.filter(~pl.col("_is_cross")).drop("_is_cross")

    out_single = out_dir / f"{target}_ic_single_asset.csv"
    out_cross = out_dir / f"{target}_ic_cross_asset.csv"
    single.write_csv(out_single)
    cross.write_csv(out_cross)
    return single.height, cross.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/ic_dashboard")
    ap.add_argument("--targets", nargs="+", default=["ES", "NQ", "RTY", "YM"])
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    for tgt in args.targets:
        in_path = in_dir / f"{tgt}_ic_2020_2023.csv"
        if not in_path.exists():
            print(f"  [skip] {in_path} not found")
            continue
        n_single, n_cross = split_one(in_path, in_dir, tgt)
        print(f"  {tgt}: single={n_single}, cross={n_cross}")


if __name__ == "__main__":
    main()
