#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=anlz_lbl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/N/project/ksb-finance-backtesting/data/label_tuning/results/anlz-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/label_tuning/results/anlz-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Per-year stability analysis. Reads the per-instrument tuning CSV, picks the
# top-N combos, and re-runs the labeler per year (2020-2023) to detect combos
# that only look balanced by averaging across regimes.
#
# Chain after tune_labels: sbatch --dependency=afterok:$TUNE_JOB analyze_tuning.sbatch
#
#   0 → ES, 1 → NQ, 2 → RTY, 3 → YM

set -euo pipefail
echo "=== task $SLURM_ARRAY_TASK_ID of job $SLURM_JOB_ID on $(hostname) at $(date) ==="

INSTRUMENTS=(ES NQ RTY YM)
INSTR="${INSTRUMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== instrument = $INSTR ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

BARS_ROOT="${BARS_ROOT:-/N/project/ksb-finance-backtesting/data/label_tuning/bars_ohlcv}"
BARS_GLOB="${BARS_ROOT}/${INSTR}/15m/${INSTR}_*_15m.parquet"

RESULTS_DIR=/N/project/ksb-finance-backtesting/data/label_tuning/results
TUNE_CSV="${RESULTS_DIR}/${INSTR}_tune_2020_2023.csv"
OUT_CSV="${RESULTS_DIR}/${INSTR}_stability_2020_2023.csv"

TOP_N="${TOP_N:-10}"

python3 -u "$HOME/us_futures/scripts/analyze_tuning.py" \
    --instrument "$INSTR" \
    --bars-glob  "$BARS_GLOB" \
    --tune-csv   "$TUNE_CSV" \
    --top        "$TOP_N" \
    --out        "$OUT_CSV"

echo "=== task $SLURM_ARRAY_TASK_ID done at $(date) with exit $? ==="
