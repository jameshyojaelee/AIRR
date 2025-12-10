#!/bin/bash
# Slurm array launcher for unified HPO sweeps.
# Usage:
#   sbatch --array=0-4 scripts/submit_hpo_array.sh

#SBATCH --job-name=airrml-hpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG="${CONFIG_PATH:-$REPO_ROOT/configs/hpo_unified.json}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-5}

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"
export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}

python3 scripts/hpo_sweep.py --config "$CONFIG" --trial-offset "$ARRAY_ID" --trial-count "$TRIALS_PER_TASK"
