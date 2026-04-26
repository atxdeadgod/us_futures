#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=fut_panel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/N/project/ksb-finance-backtesting/data/features/fut_panel-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/features/fut_panel-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Build futures feature panel for one (instrument, year). Override INSTR via env.
set -euo pipefail
YEARS=(2020 2021 2022 2023 2024)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
INSTR="${INSTR:-ES}"
echo "=== futures panel  instr=$INSTR  year=$YEAR  task=$SLURM_ARRAY_TASK_ID ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/features}"
mkdir -p "$OUT_ROOT/futures"

python3 -u /N/u/gargru/BigRed200/us_futures/scripts/build_futures_panel.py \
    --instrument "$INSTR" --year "$YEAR" --out "$OUT_ROOT"
