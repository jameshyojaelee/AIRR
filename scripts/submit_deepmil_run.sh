#!/bin/bash
# Slurm launcher for running only the Deep MIL experiment

# Usage: sbatch scripts/submit_deepmil_run.sh

#SBATCH --job-name=airrml-deepmil
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Set repo root explicitly because Slurm copies this script to a spool directory
REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"

mkdir -p logs

# Allow overriding the config path; default to the deep_mil config
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/deepmil_run1.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG_PATH")
CONFIG_STEM="${CONFIG_BASENAME%.*}"
DATE_TAG=$(date +%Y%m%d-%H%M%S)
RUN_TAG=${SLURM_JOB_ID:-$DATE_TAG}-gpu
export AIRR_OUTPUT_ROOT="${AIRR_OUTPUT_ROOT:-$REPO_ROOT/outputs/${CONFIG_STEM}-${DATE_TAG}-gpu}"

# Activate conda env with working CUDA/cuDNN build (torch 2.1.2+cu121)
source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
# Force CPU execution to avoid GPU kernel issues seen previously
export CUDA_VISIBLE_DEVICES=""

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}
export PYTHONPATH="$REPO_ROOT"

python3 scripts/run_experiment.py --config "$CONFIG_PATH"
