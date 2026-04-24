#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=spx_chain
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/spx_options_chain/slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/spx_options_chain/slurm-%j.err
#SBATCH --mail-type=END,FAIL

# NOTE: this job requires a Duo push at start. The job runs on a compute node, so
# the push arrives on your phone when srun/sbatch triggers the WRDS connect.
# If Duo fails → job exits non-zero and does NOT retry. Re-submit manually after
# reviewing what happened.

set -euo pipefail
echo "=== job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
mkdir -p /N/project/ksb-finance-backtesting/data/spx_options_chain

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

START_YEAR="${START_YEAR:-2020}"
END_YEAR="${END_YEAR:-2025}"
TICKERS="${TICKERS:-SPX,SPXW,SPY}"

python3 -u "$HOME/us_futures/scripts/wrds_spx_chain_download.py" \
    --start-year "$START_YEAR" \
    --end-year   "$END_YEAR" \
    --tickers    "$TICKERS" \
    --out-root   /N/project/ksb-finance-backtesting/data/spx_options_chain

echo "=== job $SLURM_JOB_ID finished at $(date) with exit $? ==="
