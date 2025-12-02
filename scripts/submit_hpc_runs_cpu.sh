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

# Activate conda env with working torch stack
source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}
export PYTHONPATH="$REPO_ROOT"

python3 scripts/run_experiment.py --config "$CONFIG_PATH"
