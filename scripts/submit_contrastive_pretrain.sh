#!/bin/bash
# Slurm launcher for contrastive pretraining (GPU).
#
# Usage:
#   sbatch scripts/submit_contrastive_pretrain.sh
#

#SBATCH --job-name=airrml-pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs outputs/contrastive

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/contrastive_pretrain.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

echo "Starting contrastive pretraining with config: $CONFIG_PATH"
python3 scripts/run_contrastive_pretrain.py --config "$CONFIG_PATH"
echo "Done."
