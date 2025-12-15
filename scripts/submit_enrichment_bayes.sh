#!/bin/bash
# Slurm launcher for the Bayesian enrichment model.
#
# Usage:
#   sbatch scripts/submit_enrichment_bayes.sh
#

#SBATCH --job-name=airrml-bayes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/enrichment_bayes.json}"
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

echo "Starting Bayesian Enrichment with config: $CONFIG_PATH"
python3 scripts/run_experiment.py --config "$CONFIG_PATH"
echo "Done."
