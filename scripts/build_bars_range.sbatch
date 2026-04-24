#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=bld_bars
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/N/project/ksb-finance-backtesting/data/label_tuning/bars_ohlcv/slurm-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/label_tuning/bars_ohlcv/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Array job: one task per instrument.
#   0 → ES, 1 → NQ, 2 → RTY, 3 → YM
#
# Builds OHLCV 15-min bars for label tuning only. No feature enrichment.
# Output lives in data/label_tuning/bars_ohlcv/ (self-contained tuning tree).

set -euo pipefail
echo "=== task $SLURM_ARRAY_TASK_ID of job $SLURM_JOB_ID on $(hostname) at $(date) ==="

INSTRUMENTS=(ES NQ RTY YM)
INSTR="${INSTRUMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== instrument = $INSTR ==="

mkdir -p /N/project/ksb-finance-backtesting/data/label_tuning/bars_ohlcv

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

START="${START:-2020-01-01}"
END="${END:-2023-12-31}"
HORIZON="${HORIZON:-15m}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/label_tuning/bars_ohlcv}"

python3 -u "$HOME/us_futures/scripts/build_bars_range.py" \
    --instrument "$INSTR" \
    --start      "$START" \
    --end        "$END" \
    --horizon    "$HORIZON" \
    --out-root   "$OUT_ROOT"

echo "=== task $SLURM_ARRAY_TASK_ID done at $(date) with exit $? ==="
