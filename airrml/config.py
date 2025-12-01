"""
Configuration constants and competition-specific schema for AIRR-ML-25.

Keep all dataset layout assumptions and submission column names centralized
so that the rest of the codebase can remain model-agnostic.
"""
import os
from pathlib import Path
from typing import List


def _resolve_root(env_var: str, candidates: List[str]) -> Path:
    """
    Resolve a data root, preferring an env override, then the first existing
    path in candidates. This supports both Kaggle's nested layout
    (train_datasets/train_datasets) and flat copies on HPC.
    """

    override = os.environ.get(env_var)
    if override:
        return Path(override)

    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.exists():
            return cand_path

    # Fall back to the first candidate even if missing; callers validate later.
    return Path(candidates[0])


# Base data locations (prefer full nested paths on HPC/Kaggle, fallback to flat)
TRAIN_ROOT: Path = _resolve_root("AIRR_TRAIN_ROOT", ["train_datasets/train_datasets", "train_datasets"])
TEST_ROOT: Path = _resolve_root("AIRR_TEST_ROOT", ["test_datasets/test_datasets", "test_datasets"])
OUTPUT_DIR: Path = Path("outputs")
MODEL_DIR: Path = OUTPUT_DIR / "models"
SUBMISSION_DIR: Path = OUTPUT_DIR / "submissions"

# Filenames and column names used across loaders
METADATA_FILENAME: str = "metadata.csv"
REPERTOIRE_ID_COL: str = "repertoire_id"
FILENAME_COL: str = "filename"
LABEL_COL: str = "label_positive"
SEQUENCE_COLS: List[str] = ["junction_aa", "v_call", "j_call"]

# Submission schema
SUBMISSION_COLUMNS: List[str] = [
    "ID",
    "dataset",
    "label_positive_probability",
    "junction_aa",
    "v_call",
    "j_call",
]

# Placeholder values (used when a field is intentionally left blank)
PROBABILITY_PLACEHOLDER: float = -999.0
SEQUENCE_PLACEHOLDER: str = ""

# Default experiment settings
DEFAULT_NUM_FOLDS: int = 5
DEFAULT_RANDOM_SEED: int = 42
