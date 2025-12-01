"""
High-level orchestration script for AIRR-ML-25 experiments.

Loads a YAML/JSON config, trains the specified model across all datasets,
and builds a Kaggle-ready submission.csv.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is optional
    yaml = None

import pandas as pd

from airrml import config as default_config
from airrml import training
from airrml.submission import assemble_submission
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIRR-ML-25 experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_override = os.environ.get("AIRR_OUTPUT_ROOT")

    train_root = Path(cfg.get("train_root", default_config.TRAIN_ROOT))
    test_root = Path(cfg.get("test_root", default_config.TEST_ROOT))
    output_root = Path(output_override or cfg.get("output_root", default_config.OUTPUT_DIR))
    model_name = cfg.get("model_name", "kmer_logreg")
    feature_config = cfg.get("feature_config", {"use_kmers": True, "k": 3})
    model_params = cfg.get("model_params", {})
    training_cfg = cfg.get("training", {})
    submission_cfg = cfg.get("submission", {})

    if not train_root.exists():
        raise FileNotFoundError(f"train_root does not exist: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"test_root does not exist: {test_root}")

    ensure_dir(output_root)

    cv_folds = training_cfg.get("cv_folds", 5)
    random_state = training_cfg.get("random_state", 123)

    print(f"Training model '{model_name}' across datasets in {train_root}...")
    summary = training.train_all_datasets(
        train_root=str(train_root),
        model_name=model_name,
        feature_config=feature_config,
        output_root=str(output_root),
        cv_folds=cv_folds,
        random_state=random_state,
        test_root=str(test_root),
        model_params=model_params,
    )

    mean_auc = None
    if summary:
        vals = [v for v in summary.values() if v is not None]
        mean_auc = sum(vals) / len(vals) if vals else None
        print("Per-dataset AUCs:")
        for ds, auc in summary.items():
            print(f"  {ds}: {auc}")
        if mean_auc is not None:
            print(f"Mean AUC: {mean_auc}")

    top_k_sequences = submission_cfg.get("top_k_sequences", 50000)
    if output_override:
        submission_path = Path(output_root) / "submission.csv"
    else:
        submission_path = submission_cfg.get("submission_path", output_root / "submission.csv")

    print(f"Building submission to {submission_path}...")
    assemble_submission(
        train_root=str(train_root),
        test_root=str(test_root),
        model_name=model_name,
        model_output_root=str(output_root),
        top_k_sequences=top_k_sequences,
        submission_path=str(submission_path),
    )
    print(f"Submission written to: {submission_path}")

    # Persist a run-level summary CSV for quick inspection
    write_run_summary(
        output_root=Path(output_root),
        submission_path=Path(submission_path),
        model_name=model_name,
        feature_config=feature_config,
        model_params=model_params,
        training_cfg=training_cfg,
        cv_summary=summary,
    )


def write_run_summary(
    output_root: Path,
    submission_path: Path,
    model_name: str,
    feature_config: Dict[str, Any],
    model_params: Dict[str, Any],
    training_cfg: Dict[str, Any],
    cv_summary: Dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for ds_dir in sorted(output_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        metrics_path = ds_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                with metrics_path.open("r") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}

        rows.append(
            {
                "dataset": ds_dir.name,
                "model": model_name,
                "cv_mean_auc": metrics.get("cv", {}).get("mean_auc") if isinstance(metrics.get("cv"), dict) else cv_summary.get(ds_dir.name),
                "cv_std_auc": metrics.get("cv", {}).get("std_auc") if isinstance(metrics.get("cv"), dict) else None,
                "train_samples": metrics.get("train_samples"),
                "model_path": metrics.get("model_path"),
                "feature_info_path": metrics.get("feature_info_path"),
                "submission_path": str(submission_path),
                "output_root": str(output_root),
                "feature_config": feature_config,
                "model_params": model_params,
                "training": training_cfg,
            }
        )

    summary_path = output_root / "run_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Run summary written to: {summary_path}")


if __name__ == "__main__":
    main()
