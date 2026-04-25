"""Train V1 primary LightGBM 3-class classifier.

Pipeline:
  Load panel + ILP-selected feature list
  → walk-forward CV with embargo + purge (AFML Ch. 7)
  → train one LightGBM 3-class model per fold
  → report per-fold validation metrics
  → fit final model on full IS window
  → save (model, metadata) for downstream use (meta-labeling, inference)

Class encoding for LightGBM (multiclass): {-1, 0, +1} → {0, 1, 2}.

Walk-forward CV with EMBARGO + PURGE:
  - Expanding training window
  - Fixed-size validation window
  - Embargo: gap of `embargo_bars` between train and val (avoids label
    leakage where train's last labels' forward windows overlap val)
  - Purge: drop the last `purge_bars` rows of train (their forward labels
    may have been computed using info that's now in val)

Both the embargo and the purge should equal the LABEL HORIZON (T) for the
target instrument — that's the maximum window over which a single bar's
label depends on future bars. Per LABELING_V1_SUMMARY.md, V1 T values
are 4 (RTY) or 8 (ES/NQ/YM). Set --purge-embargo to T (in bars).

Usage:
    python scripts/train_lightgbm.py \
        --panel              /N/.../feature_panels/ES_features_panel_2020_2023.parquet \
        --selected-features  /N/.../ilp_selection/ES_selected_features.csv \
        --out-model          /N/.../models/ES_lightgbm_v1.txt \
        --n-folds 5 --purge-embargo 8 --num-boost-round 500
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. `pip install lightgbm` or use bhft conda env.", file=sys.stderr)
    raise


def walk_forward_splits(
    n: int, n_folds: int = 5, embargo_bars: int = 8, purge_bars: int = 8,
    min_train_bars: int = 5000,
):
    """Yield (train_idx, val_idx) for walk-forward CV with purge + embargo.

    Expanding training window. Each fold's val set is a contiguous block
    just after the train set, with an embargo gap, and the last
    `purge_bars` of the train set are dropped to prevent label leakage.
    """
    fold_size = n // (n_folds + 1)
    if fold_size < 100:
        raise ValueError(f"Series too short ({n} bars) for n_folds={n_folds}")
    for k in range(n_folds):
        train_end = (k + 1) * fold_size
        val_start = train_end + embargo_bars
        val_end = min(val_start + fold_size, n)
        if val_end - val_start < 50:
            break
        purge_end = max(min_train_bars, train_end - purge_bars)
        train_idx = np.arange(0, purge_end)
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) < min_train_bars:
            continue
        yield train_idx, val_idx


def _multiclass_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Per-fold validation metrics: log loss, top-1 accuracy, balanced accuracy."""
    eps = 1e-12
    y_true_int = y_true.astype(int)
    n = len(y_true_int)
    # Log loss (multiclass cross-entropy)
    logp = np.log(np.clip(y_pred_proba, eps, 1.0))
    log_loss = -np.mean(logp[np.arange(n), y_true_int])
    # Top-1 accuracy
    y_hat = np.argmax(y_pred_proba, axis=1)
    acc = float((y_hat == y_true_int).mean())
    # Balanced accuracy (per-class recall mean)
    classes = np.unique(y_true_int)
    recalls = []
    for c in classes:
        mask = y_true_int == c
        if mask.sum() > 0:
            recalls.append((y_hat[mask] == c).mean())
    bal_acc = float(np.mean(recalls)) if recalls else float("nan")
    return {"log_loss": float(log_loss), "accuracy": acc, "balanced_accuracy": bal_acc}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--selected-features", required=True,
                   help="Output of ilp_select_features.py")
    p.add_argument("--out-model", required=True,
                   help="Path to save the final fitted model (LightGBM .txt)")
    p.add_argument("--out-metadata", default=None,
                   help="Path to save metadata JSON (folds, params, metrics). "
                        "Default: out-model with .json suffix.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--purge-embargo", type=int, default=8,
                   help="Bars purged at end of train + bars embargoed before val. "
                        "Should equal label horizon T (8 for ES/NQ/YM, 4 for RTY).")
    p.add_argument("--num-boost-round", type=int, default=500)
    p.add_argument("--early-stopping", type=int, default=30)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--feature-fraction", type=float, default=0.9)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    panel = pl.read_parquet(args.panel).sort("ts")
    print(f"[train] panel: {panel.height:,} rows × {panel.width} cols")

    selected = pl.read_csv(args.selected_features)["feature"].to_list()
    print(f"[train] selected features: {len(selected)}")

    missing = [f for f in selected if f not in panel.columns]
    if missing:
        raise SystemExit(f"selected features missing in panel: {missing[:10]}")

    X = panel.select(selected).to_numpy().astype(np.float32)
    y_int = panel["label"].to_numpy().astype(np.int32)
    y_lgb = (y_int + 1).astype(np.int32)  # {-1,0,+1} → {0,1,2}

    feature_names = list(selected)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "seed": args.seed,
        "verbose": -1,
    }

    print(f"[train] walk-forward CV: n_folds={args.n_folds}, "
          f"purge_embargo={args.purge_embargo}")
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(walk_forward_splits(
        n=len(y_lgb), n_folds=args.n_folds,
        embargo_bars=args.purge_embargo, purge_bars=args.purge_embargo,
    )):
        # Drop rows with any NaN feature in train/val
        tr_valid = np.isfinite(X[tr_idx]).all(axis=1)
        va_valid = np.isfinite(X[va_idx]).all(axis=1)
        tr_X, tr_y = X[tr_idx][tr_valid], y_lgb[tr_idx][tr_valid]
        va_X, va_y = X[va_idx][va_valid], y_lgb[va_idx][va_valid]

        train_data = lgb.Dataset(tr_X, label=tr_y, feature_name=feature_names)
        val_data = lgb.Dataset(va_X, label=va_y, reference=train_data)

        booster = lgb.train(
            params, train_data,
            num_boost_round=args.num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
        )

        va_proba = booster.predict(va_X, num_iteration=booster.best_iteration)
        m = _multiclass_metrics(va_y, va_proba)
        m["fold"] = fold
        m["best_iteration"] = booster.best_iteration
        m["n_train"] = int(tr_X.shape[0])
        m["n_val"] = int(va_X.shape[0])
        fold_metrics.append(m)
        print(f"  fold {fold}: train={m['n_train']:,} val={m['n_val']:,} "
              f"log_loss={m['log_loss']:.4f} acc={m['accuracy']:.3f} "
              f"bal_acc={m['balanced_accuracy']:.3f} best_iter={m['best_iteration']}")

    # Final fit on full IS window
    print(f"\n[train] fitting final model on full IS data...")
    full_valid = np.isfinite(X).all(axis=1)
    full_X, full_y = X[full_valid], y_lgb[full_valid]
    full_data = lgb.Dataset(full_X, label=full_y, feature_name=feature_names)
    avg_best = int(np.mean([m["best_iteration"] for m in fold_metrics])) if fold_metrics else args.num_boost_round
    final = lgb.train(params, full_data, num_boost_round=avg_best)

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out_model))
    print(f"[train] wrote model: {out_model}")

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "panel": str(args.panel),
        "selected_features_csv": str(args.selected_features),
        "n_features": len(selected),
        "feature_names": selected,
        "n_rows_full": int(full_X.shape[0]),
        "params": params,
        "n_folds": args.n_folds,
        "purge_embargo": args.purge_embargo,
        "fold_metrics": fold_metrics,
        "final_num_boost_round": avg_best,
    }
    out_meta = Path(args.out_metadata) if args.out_metadata else out_model.with_suffix(".json")
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[train] wrote metadata: {out_meta}")

    # Summary
    log_losses = [m["log_loss"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    bal_accs = [m["balanced_accuracy"] for m in fold_metrics]
    print(f"\n[train] CV summary:")
    print(f"  log_loss mean={np.mean(log_losses):.4f} std={np.std(log_losses):.4f}")
    print(f"  accuracy mean={np.mean(accs):.3f} std={np.std(accs):.3f}")
    print(f"  bal_acc  mean={np.mean(bal_accs):.3f} std={np.std(bal_accs):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
