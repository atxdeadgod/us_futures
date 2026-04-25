#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=feat_panel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/N/project/ksb-finance-backtesting/data/feature_panels/slurm-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/feature_panels/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Build per-target feature panel. Array per trading instrument.
set -euo pipefail
TARGETS=(ES NQ RTY YM)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "=== feature panel: target=$TARGET task=$SLURM_ARRAY_TASK_ID host=$(hostname) ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/feature_panels

START="${START:-2020-01-01}"
END="${END:-2023-12-31}"
PA_ROOT="${PA_ROOT:-/N/project/ksb-finance-backtesting/data/bars_phase_a}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/feature_panels}"

python3 -u "$HOME/us_futures/scripts/build_features_panel.py" \
    --target "$TARGET" \
    --bars-phase-a-root "$PA_ROOT" \
    --start "$START" --end "$END" \
    --out-root "$OUT_ROOT"
