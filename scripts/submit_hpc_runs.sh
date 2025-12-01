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
)

CONFIG_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"
DATE_TAG=$(date +%Y%m%d-%H%M%S)
RUN_TAG=${SLURM_JOB_ID:-$DATE_TAG}-gpu
export AIRR_OUTPUT_ROOT="$REPO_ROOT/outputs/${CONFIG_NAME%.json}-${DATE_TAG}-gpu"

# Try common module stacks; ignore if unavailable
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || module load python 2>/dev/null || true
module load CUDA/11.7.0 2>/dev/null || module load CUDA/12.1.1 2>/dev/null || module load cuda 2>/dev/null || true
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 2>/dev/null || module load PyTorch/2.1.2-foss-2023a 2>/dev/null || module load PyTorch 2>/dev/null || true
module load cudnn 2>/dev/null || true
module load gcc/11 2>/dev/null || true
module load anaconda 2>/dev/null || true
module load miniconda 2>/dev/null || true

if [ -d "$HOME/.venvs/airrml" ]; then
  source "$HOME/.venvs/airrml/bin/activate"
elif command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate airrml 2>/dev/null || true
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate airrml 2>/dev/null || true
fi

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}
export PYTHONPATH="$REPO_ROOT"

python3 scripts/run_experiment.py --config "$CONFIG_PATH"
