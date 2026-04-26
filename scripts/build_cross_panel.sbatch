#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=cross_panel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=/N/project/ksb-finance-backtesting/data/features/cross_panel-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/features/cross_panel-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Build cross-sectional panel for {target × year}. Array indexes 4 trading
# instruments × 5 years = 20 tasks.
set -euo pipefail
TARGETS=(ES NQ RTY YM)
YEARS=(2020 2021 2022 2023 2024)
TARGET=${TARGETS[$((SLURM_ARRAY_TASK_ID % 4))]}
YEAR=${YEARS[$((SLURM_ARRAY_TASK_ID / 4))]}
echo "=== cross panel  target=$TARGET  year=$YEAR  task=$SLURM_ARRAY_TASK_ID ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/features}"
mkdir -p "$OUT_ROOT/cross"

# Bound polars memory footprint for the 30-instrument wide-join. Without
# these, polars' streaming engine spins up many parallel tasks each holding
# working buffers, OOM-killing the job at 128 GB allocated.
export POLARS_MAX_THREADS=1
export POLARS_STREAMING_CHUNK_SIZE=10000

python3 -u /N/u/gargru/BigRed200/us_futures/scripts/build_cross_panel.py \
    --target "$TARGET" --year "$YEAR" --out "$OUT_ROOT"
