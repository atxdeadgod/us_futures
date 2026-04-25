#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=opt_panel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=/N/project/ksb-finance-backtesting/data/features/opt_panel-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/features/opt_panel-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Build options-derived feature panel (GEX) for one (instrument, year).
set -euo pipefail
YEARS=(2020 2021 2022 2023 2024)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
INSTR="${INSTR:-ES}"
echo "=== options panel  instr=$INSTR  year=$YEAR  task=$SLURM_ARRAY_TASK_ID ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/features}"
mkdir -p "$OUT_ROOT/options"

python3 -u "$HOME/us_futures/scripts/build_options_panel.py" \
    --instrument "$INSTR" --year "$YEAR" --out "$OUT_ROOT"
