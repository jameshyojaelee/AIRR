#!/bin/bash
# Slurm launcher for transformer Deep MIL with seed sweeps via job arrays.
#
# Usage: sbatch --array=0-2 scripts/submit_deepmil_transformer.sh

#SBATCH --job-name=airrml-dmil-tx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/commons/home/jameslee/AIRR}"
cd "$REPO_ROOT"
mkdir -p logs

BASE_CONFIG="${CONFIG_PATH:-$REPO_ROOT/configs/deepmil_transformer.json}"
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

# Seeds to sweep; override with SEEDS env (space-separated).
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 1337 2025}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$TASK_ID]:-42}

DATE_TAG=$(date +%Y%m%d-%H%M%S)
RUN_TAG="${DATE_TAG}-seed${SEED}-gpu"

TMP_CONFIG="/tmp/${USER}-dmil-tx-${SLURM_JOB_ID:-$$}-${TASK_ID}.json"

# Activate env
source /nfs/sw/easybuild/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate airrml
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

export AIRR_TRAIN_ROOT=${AIRR_TRAIN_ROOT:-/gpfs/commons/home/jameslee/AIRR/train_datasets}
export AIRR_TEST_ROOT=${AIRR_TEST_ROOT:-/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets}
export PYTHONPATH="$REPO_ROOT"

# Write a temp config with updated seed and output_root
python3 - <<PY
import json, pathlib
base = pathlib.Path("${BASE_CONFIG}")
tmp = pathlib.Path("${TMP_CONFIG}")
run_tag = "${RUN_TAG}"
seed = int(${SEED})
with base.open() as f:
    cfg = json.load(f)
cfg.setdefault("training", {})["random_state"] = seed
cfg["output_root"] = str(base.parent.parent / "outputs" / f"deepmil_transformer-{run_tag}")
tmp.write_text(json.dumps(cfg, indent=2))
print(f"Temp config written to {tmp}")
PY

AIRR_OUTPUT_ROOT="$(python3 - <<PY
import json, pathlib
tmp = pathlib.Path("${TMP_CONFIG}")
cfg = json.loads(tmp.read_text())
print(cfg.get("output_root", "outputs"))
PY
)"
export AIRR_OUTPUT_ROOT

python3 scripts/run_experiment.py --config "$TMP_CONFIG"
