#!/bin/bash
# Slurm launcher for the enrichment model (CPU-friendly).
#
# Usage:
#   sbatch scripts/submit_enrichment_run.sh
#
# Optional:
#   CONFIG_PATH=/path/to/config.json sbatch scripts/submit_enrichment_run.sh

#SBATCH --job-name=airrml-enrich
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/enrichment_run1.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

DATE_TAG=$(date +%Y%m%d-%H%M%S)
CONFIG_BASENAME=$(basename "$CONFIG_PATH")
CONFIG_STEM="${CONFIG_BASENAME%.*}"
export AIRR_OUTPUT_ROOT="${AIRR_OUTPUT_ROOT:-$REPO_ROOT/outputs/${CONFIG_STEM}-${DATE_TAG}-cpu}"

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}

python3 scripts/run_experiment.py --config "$CONFIG_PATH"

