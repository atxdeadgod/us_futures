#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=sgl_panel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=/N/project/ksb-finance-backtesting/data/features/sgl_panel-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/features/sgl_panel-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Merge futures + options panels and apply V1 labels for one (instrument, year).
# Depends on futures and options panels being built (run those arrays first).
set -euo pipefail
YEARS=(2020 2021 2022 2023 2024)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
INSTR="${INSTR:-ES}"
echo "=== single panel  instr=$INSTR  year=$YEAR  task=$SLURM_ARRAY_TASK_ID ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/features}"
mkdir -p "$OUT_ROOT/single"

python3 -u "$HOME/us_futures/scripts/build_single_panel.py" \
    --instrument "$INSTR" --year "$YEAR" --out "$OUT_ROOT"
