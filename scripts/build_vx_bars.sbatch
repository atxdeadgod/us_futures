#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=vx_bars
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/bars_phase_a/vx-slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/bars_phase_a/vx-slurm-%j.err
#SBATCH --mail-type=END,FAIL

# Build VX1/VX2/VX3 Phase A 15-min bars from raw VIX TAQ. Single task: VX has
# only 3 slots × ~1300 days, fits comfortably in one job.
set -euo pipefail

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/bars_phase_a/VX1/15m
mkdir -p /N/project/ksb-finance-backtesting/data/bars_phase_a/VX2/15m
mkdir -p /N/project/ksb-finance-backtesting/data/bars_phase_a/VX3/15m

START="${START:-2020-01-01}"
END="${END:-2024-12-31}"
HORIZON="${HORIZON:-15m}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/bars_phase_a}"
ALGOSEEK_ROOT="${ALGOSEEK_ROOT:-/N/project/ksb-finance-backtesting/data/algoseek_futures}"

python3 -u "$HOME/us_futures/scripts/build_vx_bars.py" \
    --start "$START" --end "$END" \
    --horizon "$HORIZON" \
    --out-root "$OUT_ROOT" \
    --algoseek-root "$ALGOSEEK_ROOT"
