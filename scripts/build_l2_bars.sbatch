#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=l2_bars
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/N/project/ksb-finance-backtesting/data/bars_phase_ab/slurm-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/bars_phase_ab/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Track B: enrich Phase A bars with L1-L10 depth snapshots.
set -euo pipefail
INSTRUMENTS=(ES NQ RTY YM)
INSTR=${INSTRUMENTS[$SLURM_ARRAY_TASK_ID]}
echo "=== L2 bars: $INSTR  task=$SLURM_ARRAY_TASK_ID  host=$(hostname) ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/bars_phase_ab

START="${START:-2020-01-01}"
END="${END:-2024-12-31}"
PA_ROOT="${PA_ROOT:-/N/project/ksb-finance-backtesting/data/bars_phase_a}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/bars_phase_ab}"

python3 -u "$HOME/us_futures/scripts/build_l2_bars.py" \
    --instrument "$INSTR" \
    --start "$START" --end "$END" \
    --bars-phase-a-root "$PA_ROOT" \
    --out-root "$OUT_ROOT"
