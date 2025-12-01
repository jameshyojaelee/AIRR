#!/bin/bash
# Slurm array launcher for AIRR-ML classical models on CPU (kmer, gbm)

# Usage: sbatch scripts/submit_hpc_runs_cpu.sh

#SBATCH --job-name=airrml-cpu-array
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A-%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"

mkdir -p logs

CONFIG_DIR="$REPO_ROOT/configs"
RUNS=(
  "kmer_run1.json"
  "gbm_run1.json"
)

CONFIG_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"
DATE_TAG=$(date +%Y%m%d-%H%M%S)
RUN_TAG=${SLURM_JOB_ID:-$DATE_TAG}-cpu
export AIRR_OUTPUT_ROOT="$REPO_ROOT/outputs/${CONFIG_NAME%.json}-${DATE_TAG}-cpu"

module load python 2>/dev/null || true
module load python/3.10 2>/dev/null || true
module load anaconda 2>/dev/null || true
module load miniconda 2>/dev/null || true
module load gcc/11 2>/dev/null || true

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
