#!/bin/bash
#SBATCH --job-name=airrml-gbm-arr
#SBATCH --array=0-7
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/gbm_publicness.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Config stem for consistent output separation
CONFIG_BASENAME=$(basename "$CONFIG_PATH")
CONFIG_STEM="${CONFIG_BASENAME%.*}"
# Fixed timestamp for all array jobs so they write to the same parent dir
DATE_TAG=$(date +%Y%m%d-%H%M%S) 
# Use SLURM_ARRAY_JOB_ID if available to group them, or fallback
JOB_TAG=${SLURM_ARRAY_JOB_ID:-$DATE_TAG}

export AIRR_OUTPUT_ROOT="${AIRR_OUTPUT_ROOT:-$REPO_ROOT/outputs/${CONFIG_STEM}-${JOB_TAG}-cpu}"

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

# Map array index 0-7 to dataset name 1-8
DS_IDX=$((SLURM_ARRAY_TASK_ID + 1))
DATASET_NAME="train_dataset_${DS_IDX}"

echo "Starting GBM Publicness ARRAY job for $DATASET_NAME"
echo "Config: $CONFIG_PATH"
echo "Output Root: $AIRR_OUTPUT_ROOT"

python3 scripts/run_experiment.py \
    --config "$CONFIG_PATH" \
    --dataset "$DATASET_NAME" \
    --skip-submission

echo "Done with $DATASET_NAME"
