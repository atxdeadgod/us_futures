"""How do liquidity features rank within each target's single-asset list?"""
from __future__ import annotations
from pathlib import Path
import polars as pl

TARGETS = ["ES", "NQ", "RTY", "YM"]
IC_DIR = Path("data/ic_dashboard")

# Liquidity feature patterns (cost/depth/illiquidity/concentration)
LIQ_PATTERNS = (
    "eff_spread", "depth_weighted_spread", "amihud_illiq",
    "volume_imbalance_L", "order_count_imbalance_L", "order_size_imbalance_L",
    "large_trade", "hhi_", "dw_imbalance", "cum_imbalance",
    "side_cond_", "spread", "kyle_lambda", "average_trade_size",
    "rolling_volume", "log_volume",
)


def is_liq(name: str) -> bool:
    return any(p in name for p in LIQ_PATTERNS)


for t in TARGETS:
    df = (
        pl.read_csv(IC_DIR / f"{t}_ic_single_asset.csv")
        .filter(pl.col("ic_label_spearman").is_not_null() & pl.col("ic_label_spearman").is_finite())
        .with_columns(pl.col("ic_label_spearman").abs().alias("absic"))
        .sort("absic", descending=True)
        .with_row_index("rank", offset=1)
    )
    liq = df.filter(pl.col("feature").map_elements(is_liq, return_dtype=pl.Boolean))
    print(f"\n=== {t} — liquidity features in top-50 (out of {df.height} non-NaN) ===")
    top_liq = liq.filter(pl.col("rank") <= 50)
    for r in top_liq.iter_rows(named=True):
        print(f"  rank {r['rank']:>2d}  IC={r['ic_label_spearman']:+.4f}  {r['feature']}")
    print(f"  → {top_liq.height} liquidity features in top-50, "
          f"{liq.filter(pl.col('rank') <= 100).height} in top-100")
