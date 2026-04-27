"""Check whether single-asset features rank consistently across ES/NQ/RTY/YM.

Outputs:
  - top-N intersection across all 4 targets
  - per-feature: mean |IC|, std |IC|, sign-stability across targets
  - rank-correlation of |IC| ordering between target pairs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

IC_DIR = Path("data/ic_dashboard")
TARGETS = ["ES", "NQ", "RTY", "YM"]


def load() -> dict[str, pl.DataFrame]:
    out = {}
    for t in TARGETS:
        df = pl.read_csv(IC_DIR / f"{t}_ic_single_asset.csv")
        out[t] = df.select(["feature", "ic_label_spearman", "ic_ret_spearman", "n"])
    return out


def main() -> None:
    dfs = load()

    # -- 1. Top-50 intersection of features by |ic_label_spearman| -----------
    # NaN IC means insufficient valid pairs; exclude before ranking.
    top50 = {}
    for t, df in dfs.items():
        ranked = (
            df.filter(pl.col("ic_label_spearman").is_not_null() & pl.col("ic_label_spearman").is_finite())
              .with_columns(pl.col("ic_label_spearman").abs().alias("absic"))
              .sort("absic", descending=True)
        )
        top50[t] = set(ranked.head(50)["feature"].to_list())

    inter_all = set.intersection(*top50.values())
    print(f"\n=== Top-50 |IC_label| intersection across all 4 targets: {len(inter_all)} features ===")
    for f in sorted(inter_all):
        ics = [dfs[t].filter(pl.col("feature") == f)["ic_label_spearman"][0] for t in TARGETS]
        print(f"  {f:<45s} " + " ".join(f"{t}={ic:+.4f}" for t, ic in zip(TARGETS, ics)))

    # -- 2. Pairwise top-50 overlap ------------------------------------------
    print("\n=== Pairwise top-50 overlap (single-asset features) ===")
    print("       " + "".join(f"{t:>8s}" for t in TARGETS))
    for a in TARGETS:
        row = f"{a:<6s} "
        for b in TARGETS:
            row += f"{len(top50[a] & top50[b]):>8d}"
        print(row)

    # -- 3. Per-feature consistency: only features present in all 4 ---------
    common = set.intersection(*[set(df["feature"].to_list()) for df in dfs.values()])
    rows = []
    for f in common:
        ics_lab = [dfs[t].filter(pl.col("feature") == f)["ic_label_spearman"][0] for t in TARGETS]
        ics_lab = [x for x in ics_lab if x is not None and not np.isnan(x)]
        if len(ics_lab) < 4:
            continue
        rows.append({
            "feature": f,
            "ic_ES": ics_lab[0], "ic_NQ": ics_lab[1],
            "ic_RTY": ics_lab[2], "ic_YM": ics_lab[3],
            "abs_mean": float(np.mean(np.abs(ics_lab))),
            "abs_std":  float(np.std(np.abs(ics_lab))),
            "sign_stable": all(s > 0 for s in ics_lab) or all(s < 0 for s in ics_lab),
        })
    cons = (
        pl.DataFrame(rows)
        .sort("abs_mean", descending=True)
    )

    print("\n=== Top 30 single-asset features by mean(|IC_label|) across all 4 targets ===")
    print(f"{'feature':<48s} {'ES':>8s} {'NQ':>8s} {'RTY':>8s} {'YM':>8s} {'mean':>7s} {'std':>7s} {'sign':>5s}")
    for r in cons.head(30).iter_rows(named=True):
        print(
            f"{r['feature']:<48s} "
            f"{r['ic_ES']:+8.4f} {r['ic_NQ']:+8.4f} {r['ic_RTY']:+8.4f} {r['ic_YM']:+8.4f} "
            f"{r['abs_mean']:7.4f} {r['abs_std']:7.4f} {'Y' if r['sign_stable'] else 'N':>5s}"
        )

    # -- 4. Rank correlation of |IC| ordering between target pairs ----------
    print("\n=== Spearman rank-corr of |IC_label| ordering across targets ===")
    print("       " + "".join(f"{t:>8s}" for t in TARGETS))
    for a in TARGETS:
        row = f"{a:<6s} "
        for b in TARGETS:
            ic_a = cons[f"ic_{a}"].abs().to_numpy()
            ic_b = cons[f"ic_{b}"].abs().to_numpy()
            rho, _ = spearmanr(ic_a, ic_b)
            row += f"{rho:>8.3f}"
        print(row)

    # Save consistency table for inspection
    cons.write_csv(IC_DIR / "single_asset_consistency.csv")
    print(f"\n[saved] {IC_DIR / 'single_asset_consistency.csv'} ({cons.height} features)")

    # -- 5. Sign-stable subset summary --------------------------------------
    sign_stable = cons.filter(pl.col("sign_stable"))
    print(f"\n=== Sign-stable across all 4 targets: {sign_stable.height} / {cons.height} features ===")
    print(f"  Top 15 by mean |IC|:")
    for r in sign_stable.head(15).iter_rows(named=True):
        print(f"  {r['feature']:<48s} mean|IC|={r['abs_mean']:.4f} std={r['abs_std']:.4f}")


if __name__ == "__main__":
    main()
