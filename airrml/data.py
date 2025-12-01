"""
Data loading utilities for the AIRR-ML-25 competition.

Provides filesystem discovery, metadata parsing, and repertoire-level table
construction that match the Kaggle dataset layout without relying on the
organizer's template classes.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from airrml import config

try:  # Optional torch dependency for deep models
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - placeholder for environments without torch
    class Dataset:  # type: ignore
        ...


class RepertoireDataset(Dataset):
    """
    Minimal torch Dataset wrapper around repertoire-level examples.

    This is intentionally left unimplemented here; deep models can subclass it
    once sequence preprocessing is defined.
    """

    def __init__(
        self,
        sequences: pd.DataFrame,
        metadata: pd.DataFrame,
        sequence_cols: Optional[List[str]] = None,
    ) -> None:
        self.sequences = sequences
        self.metadata = metadata
        self.sequence_cols = sequence_cols or config.SEQUENCE_COLS

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def list_datasets(train_root: Path = config.TRAIN_ROOT, test_root: Path = config.TEST_ROOT) -> Dict[str, Dict[str, Any]]:
    """
    Scan train_root for train_dataset_* folders and pair them with test_dataset_* folders.

    Supports nested layouts (e.g., */train_datasets/train_datasets) by scanning
    both the provided root and one nested level if needed.
    """

    def _collect_dirs(root: Path, prefix: str) -> List[Path]:
        direct = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        if direct:
            return sorted(direct)
        # Handle wrappers like root/test_datasets/test_dataset_*
        for wrapper in (root / "test_datasets", root / "train_datasets"):
            if wrapper.exists() and wrapper.is_dir():
                nested_dirs = [d for d in wrapper.iterdir() if d.is_dir() and d.name.startswith(prefix)]
                if nested_dirs:
                    return sorted(nested_dirs)
        # Last resort: recurse one level down to find prefixed directories
        return sorted([d for d in root.rglob(f"{prefix}*") if d.is_dir() and d.name.startswith(prefix)])

    train_root = Path(train_root)
    test_root = Path(test_root)

    if not train_root.exists():
        raise FileNotFoundError(f"Train root directory not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test root directory not found: {test_root}")

    dataset_map: Dict[str, Dict[str, Any]] = {}
    train_dirs = _collect_dirs(train_root, "train_dataset_")

    for train_dir in train_dirs:
        dataset_name = train_dir.name
        dataset_id = dataset_name.replace("train_dataset_", "")
        test_prefix = f"test_dataset_{dataset_id}"
        test_dirs = _collect_dirs(test_root, test_prefix)
        dataset_map[dataset_name] = {"train_path": train_dir, "test_paths": test_dirs}

    return dataset_map


def load_metadata(train_dataset_path: Path) -> pd.DataFrame:
    """
    Read metadata.csv from a training dataset directory and validate required columns.
    """
    metadata_path = Path(train_dataset_path) / config.METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    df = pd.read_csv(metadata_path)
    required_cols = {config.REPERTOIRE_ID_COL, config.FILENAME_COL, config.LABEL_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata at {metadata_path} is missing required columns: {missing}")

    return df


def load_repertoire_tsv(path: Path) -> pd.DataFrame:
    """
    Read a single repertoire TSV and validate required sequence columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Repertoire file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required_cols = set(config.SEQUENCE_COLS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Repertoire file {path} is missing required columns: {missing}")

    return df


def load_full_dataset(train_dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all repertoires within a training dataset into concatenated sequence and metadata tables.

    Returns:
        sequences_df: sequence-level table with columns SEQUENCE_COLS + ["ID", "label_positive"].
        metadata_df: repertoire-level table with columns ["ID", "label_positive", ...].
    """
    metadata_df = load_metadata(train_dataset_path)

    all_sequences: List[pd.DataFrame] = []
    for _, row in metadata_df.iterrows():
        filename = row[config.FILENAME_COL]
        repertoire_id = row[config.REPERTOIRE_ID_COL]
        label = row[config.LABEL_COL]

        tsv_path = Path(train_dataset_path) / filename
        rep_df = load_repertoire_tsv(tsv_path)
        rep_df = rep_df.copy()
        rep_df["ID"] = repertoire_id
        rep_df[config.LABEL_COL] = label
        all_sequences.append(rep_df)

    sequences_df = pd.concat(all_sequences, ignore_index=True) if all_sequences else pd.DataFrame()

    repertoire_metadata = metadata_df.rename(columns={config.REPERTOIRE_ID_COL: "ID"}).copy()
    repertoire_metadata = repertoire_metadata.drop(columns=[config.FILENAME_COL], errors="ignore")

    return sequences_df, repertoire_metadata


def load_test_dataset(test_dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all repertoires within a test dataset folder.

    Returns:
        sequences_df: sequence-level table with columns SEQUENCE_COLS + ["ID"].
        metadata_df: repertoire-level table with a single column "ID".
    """
    test_dataset_path = Path(test_dataset_path)
    tsv_files = sorted(test_dataset_path.glob("*.tsv"))

    all_sequences: List[pd.DataFrame] = []
    metadata_records: List[Dict[str, str]] = []

    for tsv_path in tsv_files:
        repertoire_id = tsv_path.stem
        rep_df = load_repertoire_tsv(tsv_path)
        rep_df = rep_df.copy()
        rep_df["ID"] = repertoire_id
        all_sequences.append(rep_df)
        metadata_records.append({"ID": repertoire_id})

    sequences_df = pd.concat(all_sequences, ignore_index=True) if all_sequences else pd.DataFrame()
    metadata_df = pd.DataFrame(metadata_records) if metadata_records else pd.DataFrame(columns=["ID"])

    return sequences_df, metadata_df
