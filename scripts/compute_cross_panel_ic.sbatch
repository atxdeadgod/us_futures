#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=ic_dash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/ic_dashboard/ic-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/ic_dashboard/ic-%j.err
#SBATCH --mail-type=END,FAIL

# Compute IC + t-stat dashboard for cross panel features (2020-2023 IS only).
# Override TARGET via env: TARGET=NQ sbatch ...sbatch
set -euo pipefail
TARGET="${TARGET:-ES}"

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

CROSS_ROOT="${CROSS_ROOT:-/N/project/ksb-finance-backtesting/data/features/cross}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/ic_dashboard}"
mkdir -p "$OUT_ROOT"

export POLARS_MAX_THREADS=2

python3 -u /N/u/gargru/BigRed200/us_futures/scripts/compute_cross_panel_ic.py \
    --target "$TARGET" \
    --cross-root "$CROSS_ROOT" \
    --out-root "$OUT_ROOT"
