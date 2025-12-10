"""
Pretrain a transformer encoder on sequences with contrastive InfoNCE loss.

Usage:
  python3 scripts/run_contrastive_pretrain.py --config configs/contrastive_pretrain.json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

import pandas as pd

from airrml import config as default_config
from airrml import data
from airrml.contrastive import save_state_dict, train_contrastive
from airrml.utils import ensure_dir


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs")
        with path.open("r") as f:
            return yaml.safe_load(f)
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive pretraining for sequences")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_root = Path(cfg.get("train_root", default_config.TRAIN_ROOT))
    datasets = cfg.get("datasets", [])
    out_path = Path(cfg.get("output_path", "outputs/contrastive/encoder.pt"))
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentations", {})

    # Collect sequences from selected datasets (all if empty list)
    dataset_map = data.list_datasets(train_root, default_config.TEST_ROOT)
    selected = {k: v for k, v in dataset_map.items() if not datasets or k in datasets}
    seqs: list[str] = []
    for _, info in selected.items():
        seq_df, _ = data.load_full_dataset(info["train_path"])
        seqs.extend(seq_df["junction_aa"].dropna().astype(str).tolist())

    if not seqs:
        raise ValueError("No sequences found for contrastive pretraining.")

    state = train_contrastive(
        sequences=seqs,
        model_dim=model_cfg.get("model_dim", 256),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 4),
        ff_dim=model_cfg.get("ff_dim"),
        dropout=model_cfg.get("dropout", 0.1),
        max_len=model_cfg.get("max_len", 72),
        batch_size=train_cfg.get("batch_size", 256),
        num_epochs=train_cfg.get("num_epochs", 5),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        aug_cfg=aug_cfg,
        seed=train_cfg.get("random_state", 42),
        device=train_cfg.get("device"),
    )

    ensure_dir(out_path.parent)
    save_state_dict(state, out_path)
    print(f"Saved pretrained encoder to {out_path}")


if __name__ == "__main__":
    main()
