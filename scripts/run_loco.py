"""
Run leave-one-dataset-out validation for AIRR-ML models.

Usage:
  python3 scripts/run_loco.py --config configs/deepmil_transformer.json

Outputs a CSV with per-holdout AUCs and prints mean/std.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is optional
    yaml = None

import numpy as np
import pandas as pd

from airrml import config as default_config
from airrml import pipeline
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
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out validation")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_root = Path(cfg.get("train_root", default_config.TRAIN_ROOT))
    test_root = Path(cfg.get("test_root", default_config.TEST_ROOT))
    feature_config = cfg.get("feature_config", {"use_kmers": True, "k": 3})
    model_name = cfg.get("model_name", "deep_mil")
    model_params = cfg.get("model_params", {})
    training_cfg = cfg.get("training", {"random_state": default_config.DEFAULT_RANDOM_SEED})
    # Pass model_params/test_root through training_config for downstream functions
    training_cfg = dict(training_cfg)
    training_cfg["model_params"] = model_params
    training_cfg["test_root"] = str(test_root)

    results = pipeline.cross_validate_across_datasets(
        model_name=model_name,
        train_root=train_root,
        feature_config=feature_config,
        training_config=training_cfg,
    )

    rows = []
    aucs = []
    for ds, metrics in results.items():
        auc_val = metrics.get("holdout_auc")
        rows.append({"dataset": ds, "holdout_auc": auc_val})
        if auc_val is not None and not np.isnan(auc_val):
            aucs.append(auc_val)

    loco_cfg = cfg.get("loco", {})
    out_path = Path(loco_cfg.get("output_path", Path(cfg.get("output_root", "outputs/loco")) / "loco_summary.csv"))
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    std_auc = float(np.std(aucs)) if aucs else float("nan")
    print(f"LOCO results written to: {out_path}")
    print(f"Mean AUC: {mean_auc:.4f}, Std AUC: {std_auc:.4f}")


if __name__ == "__main__":
    main()
