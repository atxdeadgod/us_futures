"""Build the complete ES single-contract feature panel.

Reads:
    Phase A+B 15m bars     /N/.../bars_phase_ab/ES/15m/ES_{YYYYMMDD}_15m.parquet
    VX1/VX2/VX3 15m bars   /N/.../bars_phase_a/VX{i}/15m/VX{i}_{YYYYMMDD}_15m.parquet
    GEX daily profile      /N/.../gex_features/SPX_{YEAR}.parquet  (per year)

Pipeline (ES-only, no cross-asset):
    1. Concat per-day parquets across requested date range → continuous panel
    2. attach_base_microstructure_features  (40+ per-bar values)
    3. attach_session_flags + attach_minute_of_day_cyclic
    4. attach_overnight_features
    5. attach_pattern_features  (Tier-7 detectors)
    6. attach_engine_features   (fracdiff, round-number pins)
    7. attach_l2_deep_features  (Phase A+B per-level + composite)
    8. attach_vx_features        (asof-join VX1/VX2/VX3 → vx mid/spread/curvature)
    9. attach_gex_for_target     (SPX → ES, optional)
    10. attach_ts_normalizations on BASE_VALUE_COLS  (TC + MAD z-scores × multiple windows)
    11. attach_ema_smoothed       (causal EMA of key features)
    12. assemble_target_panel    (apply triple-barrier labels, drop warmup)
    13. Write parquet            /N/.../features/ES_panel_{YYYY}.parquet

Usage:
    python scripts/build_es_panel.py --year 2024 --out /N/.../features/
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features import panel


def _stitch_per_day_parquets(root: Path, instr: str, horizon: str,
                             year: int) -> pl.DataFrame:
    """Concat all per-day parquets for (instr, horizon) within `year`."""
    folder = root / instr / horizon
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = sorted(folder.glob(f"{instr}_{year}*_{horizon}.parquet"))
    if not files:
        raise FileNotFoundError(f"No {instr} parquets in {folder} for year {year}")
    return pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed").sort("ts")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", required=True, help="Output directory for the panel parquet")
    p.add_argument("--bars-phase-a-root", default="/N/project/ksb-finance-backtesting/data/bars_phase_a")
    p.add_argument("--bars-phase-ab-root", default="/N/project/ksb-finance-backtesting/data/bars_phase_ab")
    p.add_argument("--gex-root", default="/N/project/ksb-finance-backtesting/data/gex_features")
    p.add_argument("--horizon", default="15m")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--label", action="store_true", help="Apply triple-barrier labels")
    p.add_argument("--no-l2-deep", action="store_true",
                   help="Skip L2-deep features (use Phase A only — won't have L1-L10 cols)")
    p.add_argument("--no-vx", action="store_true")
    p.add_argument("--no-gex", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    bars_phase_a = Path(args.bars_phase_a_root)
    bars_phase_ab = Path(args.bars_phase_ab_root)
    gex_root = Path(args.gex_root)

    # Use Phase A+B for the trading instrument so L2-deep is available.
    src_root = bars_phase_a if args.no_l2_deep else bars_phase_ab
    print(f"[load] {args.instrument} {args.horizon} from {src_root}")
    bars = _stitch_per_day_parquets(src_root, args.instrument, args.horizon, args.year)
    print(f"  loaded {bars.height:,} bars  cols={len(bars.columns)}")

    # Step 1-6: per-instrument core feature pass (base + session + cyclic +
    # overnight + patterns + engines + ts-normalizations + EMA-smoothed)
    print("[features] per-instrument core pass")
    feat = panel.build_per_instrument_features(
        bars,
        lookback_days_grid=(30, 60),
        attach_overnight=True,
        attach_patterns=True,
        attach_engines=True,
        attach_smoothed=True,
        attach_cyclic_minute=True,
    )
    print(f"  after core pass: cols={len(feat.columns)}")

    # Step 7: L2-deep (only if Phase A+B)
    if not args.no_l2_deep:
        print("[features] L2 deep + per-level")
        feat = panel.attach_l2_deep_features(feat, depth=10, spread_z_window=60)
        print(f"  after L2 deep: cols={len(feat.columns)}")

    # Step 8: VX features (asof-join VX1/VX2/VX3)
    if not args.no_vx:
        try:
            print("[features] VX1/VX2/VX3 asof-join")
            vx1 = _stitch_per_day_parquets(bars_phase_a, "VX1", args.horizon, args.year)
            vx2 = _stitch_per_day_parquets(bars_phase_a, "VX2", args.horizon, args.year)
            vx3 = _stitch_per_day_parquets(bars_phase_a, "VX3", args.horizon, args.year)
            feat = panel.attach_vx_features(feat, vx1_bars=vx1, vx2_bars=vx2, vx3_bars=vx3)
            print(f"  after VX: cols={len(feat.columns)}")
        except FileNotFoundError as e:
            print(f"  [skip-vx] {e}")

    # Step 9: GEX features
    if not args.no_gex:
        gex_paths = sorted(gex_root.glob(f"SPX_gex_profile_{args.year}.parquet"))
        if gex_paths:
            print(f"[features] GEX SPX → ES, {len(gex_paths)} profile(s)")
            feat = panel.attach_gex_for_target(feat, [str(p) for p in gex_paths])
            print(f"  after GEX: cols={len(feat.columns)}")
        else:
            print(f"  [skip-gex] no SPX_{args.year}*.parquet in {gex_root}")

    # Step 10: optional labeling
    if args.label and args.instrument in panel.V1_LABEL_PARAMS:
        print(f"[labels] V1 triple-barrier {args.instrument}")
        feat = panel.assemble_target_panel(
            target=args.instrument,
            target_bars_with_features=feat,
            wide_cross_asset=None,  # single-contract pipeline
            drop_invalid=True,
        )
        print(f"  after labeling: rows={feat.height:,}  cols={len(feat.columns)}")

    out_path = out_root / f"{args.instrument}_panel_{args.year}.parquet"
    feat.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[done] {out_path}  rows={feat.height:,}  cols={len(feat.columns)}")
    # Print a sample of the column groups
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
