#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=vix_sync
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/N/project/ksb-finance-backtesting/data/algoseek_futures/vix/sync-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/algoseek_futures/vix/sync-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Array task per year (2020-2025).
set -euo pipefail
YEARS=(2020 2021 2022 2023 2024 2025)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
echo "=== VIX sync year=$YEAR task=$SLURM_ARRAY_TASK_ID host=$(hostname) ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/algoseek_futures/vix

python3 -u "$HOME/us_futures/scripts/sync_vix.py" \
    --year "$YEAR" \
    --threads 8 \
    --log "/N/project/ksb-finance-backtesting/data/algoseek_futures/vix/sync-${YEAR}.log"
