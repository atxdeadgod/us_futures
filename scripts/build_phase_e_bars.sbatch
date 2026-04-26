#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=phase_e
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/bars_phase_e/slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/bars_phase_e/slurm-%j.err
#SBATCH --mail-type=END,FAIL

# Build Phase E (execution-quality + cancel-proxy + quote-event aggregates) for
# one instrument's full date range. Default ES; override INSTR via env.
set -euo pipefail

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

INSTR="${INSTR:-ES}"
START="${START:-2020-01-01}"
END="${END:-2024-12-31}"
HORIZON="${HORIZON:-15m}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/bars_phase_e}"
ALGOSEEK_ROOT="${ALGOSEEK_ROOT:-/N/project/ksb-finance-backtesting/data/algoseek_futures}"

mkdir -p "$OUT_ROOT/$INSTR/$HORIZON"

echo "=== Phase E bars  instr=$INSTR  range=[$START..$END]  horizon=$HORIZON ==="

python3 -u /N/u/gargru/BigRed200/us_futures/scripts/build_phase_e_bars.py \
    --instrument "$INSTR" \
    --start "$START" --end "$END" \
    --horizon "$HORIZON" \
    --out-root "$OUT_ROOT" \
    --algoseek-root "$ALGOSEEK_ROOT"
