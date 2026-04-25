#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=es_panel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/N/project/ksb-finance-backtesting/data/features/es_panel-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/features/es_panel-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Build ES single-contract feature panel for one year per array task.
set -euo pipefail
YEARS=(2020 2021 2022 2023 2024)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
echo "=== ES panel build year=$YEAR  task=$SLURM_ARRAY_TASK_ID ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/features}"
mkdir -p "$OUT_ROOT"

python3 -u "$HOME/us_futures/scripts/build_es_panel.py" \
    --year "$YEAR" --out "$OUT_ROOT" --label
