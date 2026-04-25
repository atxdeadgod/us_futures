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

python3 -u "$HOME/us_futures/scripts/build_gex_features.py" \
    --tickers "SPX,SPY" \
    --start-year 2020 --end-year 2024 \
    --spx-chain-root "$CHAIN_ROOT" \
    --out-root "$OUT_ROOT"
