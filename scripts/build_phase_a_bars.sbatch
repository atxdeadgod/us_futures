#!/bin/bash
#SBATCH --account=r01892
#SBATCH --job-name=phase_a
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --array=0-29
#SBATCH --output=/N/project/ksb-finance-backtesting/data/bars_phase_a/slurm-%A_%a.out
#SBATCH --error=/N/project/ksb-finance-backtesting/data/bars_phase_a/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL

# Phase A bars: OHLCV + aggressor + L1 close + spread + CVD per (instrument, day),
# downsampled to 15-min. Array per instrument across the FULL 30-contract universe
# — equity indices (V1 trading), FX (DXY composite), rates, metals, energy, ag.
# Cross-asset macro features in panel.py need OHLCV-level data on all 30.
set -euo pipefail
INSTRUMENTS=(
    ES NQ RTY YM                            # equity indices (4)
    6A 6B 6C 6E 6J                          # FX (5)
    BZ CL HO NG RB                          # energy (5)
    GC HG PA PL SI                          # metals (5)
    SR3 TN ZB ZF ZN ZT                      # rates / curve (6)
    ZC ZL ZM ZS ZW                          # ags (5)
)
INSTR=${INSTRUMENTS[$SLURM_ARRAY_TASK_ID]}
echo "=== Phase A bars: $INSTR  task=$SLURM_ARRAY_TASK_ID  host=$(hostname) ==="

source /N/slate/gargru/miniforge3/etc/profile.d/conda.sh
conda activate bhft

mkdir -p /N/project/ksb-finance-backtesting/data/bars_phase_a

START="${START:-2020-01-01}"
END="${END:-2024-12-31}"
HORIZON="${HORIZON:-15m}"
OUT_ROOT="${OUT_ROOT:-/N/project/ksb-finance-backtesting/data/bars_phase_a}"

python3 -u /N/u/gargru/BigRed200/us_futures/scripts/build_phase_a_bars.py \
    --instrument "$INSTR" \
    --start "$START" --end "$END" \
    --horizon "$HORIZON" \
    --out-root "$OUT_ROOT"
