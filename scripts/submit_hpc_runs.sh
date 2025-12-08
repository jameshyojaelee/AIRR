#!/bin/bash
# Slurm array launcher for AIRR-ML experiments

# Usage: sbatch scripts/submit_hpc_runs.sh

#SBATCH --job-name=airrml-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A-%a.out

set -euo pipefail

# Set repo root explicitly because Slurm copies this script to a spool directory
REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"

mkdir -p logs

CONFIG_DIR="$REPO_ROOT/configs"
RUNS=(
  "kmer_run1.json"
  "gbm_run1.json"
  "deepmil_run1.json"
  "gbm_grid_k5_depth6.json"
  "gbm_grid_k5_depth8.json"
  "gbm_grid_k6_depth6.json"
  "gbm_grid_k6_depth8_lr003.json"
  "stacked_run1.json"
)

CONFIG_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"
DATE_TAG=$(date +%Y%m%d-%H%M%S)
RUN_TAG=${SLURM_JOB_ID:-$DATE_TAG}-gpu
export AIRR_OUTPUT_ROOT="$REPO_ROOT/outputs/${CONFIG_NAME%.json}-${DATE_TAG}-gpu"

# Activate conda env with working CUDA/cuDNN build (torch 2.1.2+cu121)
source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}
export PYTHONPATH="$REPO_ROOT"

python3 scripts/run_experiment.py --config "$CONFIG_PATH"
