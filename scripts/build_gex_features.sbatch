#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=gex_feat
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/gex_features/slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/gex_features/slurm-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
echo "=== GEX features build host=$(hostname) ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/gex_features

CHAIN_ROOT="${CHAIN_ROOT:-/N/project/ksb-finance-backtesting/data/spx_options_chain}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/gex_features}"
TICKERS="${TICKERS:-SPX,SPY}"
START_YEAR="${START_YEAR:-2020}"
END_YEAR="${END_YEAR:-2024}"

# Code lives on the BigRed200 home, but slate (where conda is) is shared, and
# /N/u/gargru/BigRed200/... is readable from quartz too — so this sbatch
# works from either cluster's submitter.
python3 -u /N/u/gargru/BigRed200/us_futures/scripts/build_gex_features.py \
    --tickers "$TICKERS" \
    --start-year "$START_YEAR" --end-year "$END_YEAR" \
    --spx-chain-root "$CHAIN_ROOT" \
    --out-root "$OUT_ROOT"
