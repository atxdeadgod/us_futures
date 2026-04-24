#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=algoseek_p2
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/N/project/ksb-finance-backtesting/data/algoseek_futures/slurm-%j.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/algoseek_futures/slurm-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
echo "=== job $SLURM_JOB_ID started on $(hostname) at $(date) ==="

# Phase 2: 24 remaining roots from DESIGN.md universe
# (excludes ES,NQ,RTY,YM,ZN,6E + VIX which are covered by phase 1)
ROOTS_P2="ZT,ZF,TN,ZB,CL,BZ,NG,HO,RB,GC,SI,HG,PA,PL,ZC,ZS,ZW,ZM,ZL,6J,6B,6A,6C,SR3"

START="${START:-2020-01-01}"
END="${END:-2025-12-31}"
THREADS="${THREADS:-32}"

echo "START=$START  END=$END  THREADS=$THREADS  ROOTS=$ROOTS_P2  VIX=skip"

python3 -u "$HOME/algoseek_hpc_sync.py" \
    --start "$START" \
    --end   "$END" \
    --threads "$THREADS" \
    --roots "$ROOTS_P2" \
    --no-vix \
    --log /N/project/ksb-finance-backtesting/data/algoseek_futures/sync_phase2.log

echo "=== job $SLURM_JOB_ID finished at $(date) with exit $? ==="
