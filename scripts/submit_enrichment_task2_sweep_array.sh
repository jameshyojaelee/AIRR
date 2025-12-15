#!/bin/bash
# Slurm array launcher for enrichment Task-2 sweep (CPU).
#
# Usage:
#   sbatch --array=0-4 scripts/submit_enrichment_task2_sweep_array.sh
#
# Optional:
#   CONFIG_PATH=/path/to/config.json TRIALS_PER_TASK=10 sbatch --array=0-7 scripts/submit_enrichment_task2_sweep_array.sh

#SBATCH --job-name=airrml-enrich-sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/enrichment_sweep_task2.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-5}

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}

python3 scripts/enrichment_task2_sweep.py --config "$CONFIG_PATH" --trial-offset "$ARRAY_ID" --trial-count "$TRIALS_PER_TASK"

