"""Build the SINGLE-CONTRACT merged panel for one (instrument, year).

Joins the per-source panels (futures + options) and applies V1 triple-barrier
labels. This is the per-target output that feeds the cross-sectional pipeline
and any single-instrument modeling.

Reads:
    {OUT_ROOT}/futures/{INSTR}_{YEAR}.parquet     (required)
    {OUT_ROOT}/options/{INSTR}_{YEAR}.parquet     (optional; skipped if absent)

Produces:
    {OUT_ROOT}/single/{INSTR}_{YEAR}.parquet

Pipeline:
    futures + options frames → left-join on ts (futures is authoritative for
    duplicate cols) → assemble_target_panel (applies labels, drops warmup).

Usage:
    python scripts/build_single_panel.py --instrument ES --year 2024 \
        --out /N/.../features [--no-label]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import labeling


def _print_col_summary(feat: pl.DataFrame) -> None:
    grouped: dict[str, list[str]] = {
        "base": [], "norm_tc_z": [], "norm_madz": [],
        "L2_deep": [], "L2_per_level": [],
        "patterns": [], "engines": [], "vx": [], "gex": [],
        "ema_smoothed": [], "session/cyclic/overnight": [], "labels/identity": [],
    }
    for c in feat.columns:
        if "_tc_z_w" in c: grouped["norm_tc_z"].append(c)
        elif "_tc_madz_w" in c: grouped["norm_madz"].append(c)
        elif c in ("hhi_bid_d10", "hhi_ask_d10", "cum_imbalance_d10", "dw_imbalance_d10",
                    "depth_weighted_spread_d10", "liquidity_adjusted_spread_d10",
                    "spread_acceleration", "deep_ofi_d10_decay0", "deep_ofi_d10_decay03",
                    "spread_zscore_w60"):
            grouped["L2_deep"].append(c)
        elif any(c.startswith(p + "_L") for p in ("volume_imbalance", "basic_spread",
                                                    "ofi_at", "order_count_imbalance",
                                                    "order_size_imbalance")):
            grouped["L2_per_level"].append(c)
        elif c.startswith("vx") or c.startswith("VX"): grouped["vx"].append(c)
        elif c.startswith("breakout_") or c.startswith("absorption_") or c.startswith("cvd_price_div") \
            or c.startswith("range_compression") or c.startswith("imbalance_persistence") \
            or c.startswith("spike_and_fade") or c.startswith("post_breakout") \
            or c == "atr_proxy":
            grouped["patterns"].append(c)
        elif c.startswith("fracdiff_") or c.startswith("round_pin_"): grouped["engines"].append(c)
        elif c.startswith("gex_") or c in ("total_gex", "gex_sign", "distance_to_zero_gamma_flip",
                                             "distance_to_max_call_oi", "distance_to_max_put_oi"):
            grouped["gex"].append(c)
        elif c.endswith("_ema_s10") or c.endswith("_ema_s30"): grouped["ema_smoothed"].append(c)
        elif c in ("hour_et", "is_asia", "is_eu", "is_rth", "is_eth",
                    "minute_of_day_sin", "minute_of_day_cos") or c.startswith("overnight_"):
            grouped["session/cyclic/overnight"].append(c)
        elif c in ("ts", "root", "expiry", "is_session_warm",
                    "label", "realized_ret", "realized_ret_pts", "hit_offset",
                    "halt_truncated", "atr"):
            grouped["labels/identity"].append(c)
        else:
            grouped["base"].append(c)
    print("\n[col-summary]")
    for k, v in grouped.items():
        print(f"  {k:30s}  n={len(v):4d}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True, help="Output root containing futures/, options/, single/ subdirs")
    p.add_argument("--no-label", action="store_true",
                   help="Skip triple-barrier labeling")
    args = p.parse_args()

    out_root = Path(args.out)
    futures_path = out_root / "futures" / f"{args.instrument}_{args.year}.parquet"
    options_path = out_root / "options" / f"{args.instrument}_{args.year}.parquet"
    single_dir = out_root / "single"
    single_dir.mkdir(parents=True, exist_ok=True)

    if not futures_path.exists():
        print(f"[error] missing futures panel: {futures_path}")
        return 1
    print(f"[load] futures: {futures_path}")
    feat = pl.read_parquet(futures_path).sort("ts")
    print(f"  rows={feat.height:,}  cols={len(feat.columns)}")

    if options_path.exists():
        print(f"[load] options: {options_path}")
        opts = pl.read_parquet(options_path).sort("ts")
        # Left-join: futures is authoritative for any duplicate ts row
        already = set(feat.columns)
        opts_unique = ["ts"] + [c for c in opts.columns if c != "ts" and c not in already]
        feat = feat.join(opts.select(opts_unique), on="ts", how="left")
        print(f"  after options join: cols={len(feat.columns)}")
    else:
        print(f"[skip] no options panel at {options_path}")

    if not args.no_label and args.instrument in labeling.V1_LABEL_PARAMS:
        print(f"[labels] V1 triple-barrier {args.instrument}")
        feat = labeling.assemble_target_panel(
            target=args.instrument,
            target_bars_with_features=feat,
            wide_cross_asset=None,  # cross-sectional join happens in build_cross_panel.py
            drop_invalid=True,
        )
        print(f"  after labeling: rows={feat.height:,}  cols={len(feat.columns)}")

    out_path = single_dir / f"{args.instrument}_{args.year}.parquet"
    feat.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[done] {out_path}  rows={feat.height:,}  cols={len(feat.columns)}")
    _print_col_summary(feat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
