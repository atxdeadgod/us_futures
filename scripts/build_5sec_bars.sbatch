#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=bars_5s
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/bars_5sec/slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/bars_5sec/slurm-%j.err
#SBATCH --mail-type=END,FAIL

# Build raw 5-sec Phase A bars for ES (single instrument; expand later if needed).
# Independent of VX bar build and ES panel build; runs in parallel.
set -euo pipefail

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/bars_5sec/ES/5s

INSTR="${INSTR:-ES}"
START="${START:-2020-01-01}"
END="${END:-2024-12-31}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/bars_5sec}"

python3 -u /N/u/gargru/BigRed200/us_futures/scripts/build_5sec_bars.py \
    --instrument "$INSTR" \
    --start "$START" --end "$END" \
    --out-root "$OUT_ROOT"
