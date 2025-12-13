#!/bin/bash
# Assemble submissions from existing deep_mil output directories (no retraining).
#
# Usage:
#   sbatch --array=0-2 scripts/submit_assemble_deepmil_outputs.sh
#   # or override dirs:
#   OUTPUT_DIRS="outputs/runA outputs/runB" sbatch --array=0-1 scripts/submit_assemble_deepmil_outputs.sh

#SBATCH --job-name=airrml-assemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}

# Resolve list of output directories
if [[ -n "${OUTPUT_DIRS:-}" ]]; then
  IFS=' ' read -r -a DIRS <<< "${OUTPUT_DIRS}"
else
  DIRS=(outputs/deepmil_transformer-*-seed*-gpu)
fi

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "No output directories found." >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_OUTPUT_ROOT="${DIRS[$TASK_ID]:-}"
if [[ -z "$MODEL_OUTPUT_ROOT" || ! -d "$MODEL_OUTPUT_ROOT" ]]; then
  echo "Invalid MODEL_OUTPUT_ROOT for task $TASK_ID: '$MODEL_OUTPUT_ROOT'" >&2
  exit 1
fi

TOP_K=${TOP_K:-50000}
SUBMISSION_PATH="${SUBMISSION_PATH:-$MODEL_OUTPUT_ROOT/submission.csv}"
export MODEL_OUTPUT_ROOT TOP_K SUBMISSION_PATH

python3 - <<PY
import os
from airrml.submission import assemble_submission

assemble_submission(
    train_root=os.environ["AIRR_TRAIN_ROOT"],
    test_root=os.environ["AIRR_TEST_ROOT"],
    model_name="deep_mil",
    model_output_root=os.environ["MODEL_OUTPUT_ROOT"],
    top_k_sequences=int(os.environ["TOP_K"]),
    submission_path=os.environ["SUBMISSION_PATH"],
)
print(f"Wrote submission to {os.environ['SUBMISSION_PATH']}")
PY
