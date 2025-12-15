#!/bin/bash
# Slurm launcher for GBM + Publicness features (CPU).
#
# Usage:
#   sbatch scripts/submit_gbm_publicness.sh
#

#SBATCH --job-name=airrml-gbm-pub
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=84:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/gbm_publicness.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG_PATH")
CONFIG_STEM="${CONFIG_BASENAME%.*}"
DATE_TAG=$(date +%Y%m%d-%H%M%S)
export AIRR_OUTPUT_ROOT="${AIRR_OUTPUT_ROOT:-$REPO_ROOT/outputs/${CONFIG_STEM}-${DATE_TAG}-cpu}"

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

echo "Starting GBM Publicness run with config: $CONFIG_PATH"
python3 scripts/run_experiment.py --config "$CONFIG_PATH"
echo "Done."
