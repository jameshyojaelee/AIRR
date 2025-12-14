#!/bin/bash
# Slurm launcher for Deep MIL fine-tuning (GPU).
#
# Usage:
#   sbatch scripts/submit_deepmil_finetune.sh
#

#SBATCH --job-name=airrml-dmil-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/deepmil_finetune_v1.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG_PATH")
CONFIG_STEM="${CONFIG_BASENAME%.*}"
DATE_TAG=$(date +%Y%m%d-%H%M%S)
export AIRR_OUTPUT_ROOT="${AIRR_OUTPUT_ROOT:-$REPO_ROOT/outputs/${CONFIG_STEM}-${DATE_TAG}-gpu}"

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

echo "Starting Deep MIL fine-tuning with config: $CONFIG_PATH"
python3 scripts/run_experiment.py --config "$CONFIG_PATH"
echo "Done."
