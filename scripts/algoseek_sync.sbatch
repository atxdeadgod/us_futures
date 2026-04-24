#!/bin/bash
#SBATCH --job-name=algoseek_sync
#SBATCH --account=r01892
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/algoseek_futures/slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/algoseek_futures/slurm-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
echo "=== job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "cores=$SLURM_CPUS_PER_TASK  mem=$SLURM_MEM_PER_NODE"

# Defaults — overridable from sbatch invocation:  sbatch --export=START=...,END=... script
START="${START:-2020-01-01}"
END="${END:-2025-12-31}"
THREADS="${THREADS:-32}"

echo "START=$START  END=$END  THREADS=$THREADS"

# AWS profile + creds use ~/.aws/credentials (plt-de-dev)
# Use system python3 (we don't need conda for this)
which python3
python3 --version

python3 -u "$HOME/algoseek_hpc_sync.py" \
    --start "$START" \
    --end   "$END" \
    --threads "$THREADS"

echo "=== job $SLURM_JOB_ID finished at $(date) with exit $? ==="
