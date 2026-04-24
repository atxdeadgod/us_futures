#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=tune_lbl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/N/project/ksb-finance-backtesting/data/label_tuning/slurm-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/label_tuning/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Array job: one task per instrument.
#   SLURM_ARRAY_TASK_ID=0 → ES
#   SLURM_ARRAY_TASK_ID=1 → NQ
#   SLURM_ARRAY_TASK_ID=2 → RTY
#   SLURM_ARRAY_TASK_ID=3 → YM
#
# IS window = 2020-2023. 2024+ is OOS and must not be tuned on.

set -euo pipefail
echo "=== array task $SLURM_ARRAY_TASK_ID of job $SLURM_JOB_ID on $(hostname) at $(date) ==="

INSTRUMENTS=(ES NQ RTY YM)
INSTR="${INSTRUMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== instrument = $INSTR ==="

mkdir -p /N/project/ksb-finance-backtesting/data/label_tuning

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

# Bars path convention: one parquet per (instrument, day) at 15-min horizon.
# Adjust BARS_ROOT if the bar-building pipeline writes elsewhere.
BARS_ROOT="${BARS_ROOT:-/N/project/ksb-finance-backtesting/data/bars}"
BARS_GLOB="${BARS_ROOT}/${INSTR}/15m/${INSTR}_*_15m.parquet"

OUT_DIR=/N/project/ksb-finance-backtesting/data/label_tuning
OUT_CSV="${OUT_DIR}/${INSTR}_tune_2020_2023.csv"

python3 -u "$HOME/us_futures/scripts/tune_labels.py" \
    --instrument "$INSTR" \
    --bars-glob  "$BARS_GLOB" \
    --start      2020-01-01 \
    --end        2023-12-31 \
    --out        "$OUT_CSV"

echo "=== array task $SLURM_ARRAY_TASK_ID done at $(date) with exit $? ==="
