"""
Training harness for AIRR-ML-25 models.

Provides dataset-wise training, cross-validation orchestration, and artifact
persistence independent of specific model implementations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from airrml import config
from airrml import data
from airrml import evaluation
from airrml.features import build_combined_feature_matrix
from airrml.models import get_model
from airrml.utils import ensure_dir

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _save_feature_info(path: Path, feature_info: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(feature_info, f, indent=2, default=str)


def train_on_dataset(
    dataset_name: str,
    train_path: str,
    model_name: str,
    feature_config: Dict[str, Any],
    output_dir: str,
    cv_folds: int = 5,
    random_state: int = 123,
    model_params: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Train a model on a single train_dataset_X and optionally run cross-validation.
    """
    train_path = Path(train_path)
    out_dir = Path(output_dir) / dataset_name
    ensure_dir(out_dir)

    sequences_df, metadata_df = data.load_full_dataset(train_path)

    # Avoid silent downsampling; ignore any legacy max_sequences entry
    if feature_config and feature_config.get("max_sequences") is not None:
        feature_config = dict(feature_config)
        feature_config.pop("max_sequences", None)
    metrics: Dict[str, Any] = {}

    # Cross-validation (if enough repertoires)
    if cv_folds and len(metadata_df) >= cv_folds:
        cv_results = evaluation.cross_validate_model(
            model_name=model_name,
            sequences_df=sequences_df,
            label_df=metadata_df,
            feature_config=feature_config,
            cv_folds=cv_folds,
            random_state=random_state,
            model_params=model_params,
        )
        metrics["cv"] = cv_results

    # Fit on full data
    model_params = model_params or {}
    model = get_model(model_name, random_state=random_state, **model_params)
    feature_info: Optional[Dict[str, Any]] = None

    if model_name == "deep_mil":
        labels = metadata_df.set_index("ID")[config.LABEL_COL]
        model.fit(sequences_df, labels)
        feature_info = getattr(model, "feature_info", None)
    else:
        X_full, y_full, feature_info = build_combined_feature_matrix(sequences_df, metadata_df, feature_config)
        model.set_feature_info(feature_info)
        model.fit(X_full, y_full)

    # Save artifacts
    model_path = out_dir / ("model.pt" if model_name == "deep_mil" else "model.joblib")
    model_params_path = out_dir / "model_params.json"
    feature_info_path = out_dir / "feature_info.json"
    metrics_path = out_dir / "metrics.json"

    # Try model-specific save if available
    saved = False
    if resume and model_path.exists():
        # Skip saving if resuming (model already on disk)
        saved = True
    if hasattr(model, "save") and not saved:
        try:
            model.save(str(model_path))
            saved = True
        except Exception:
            saved = False

    if not saved:
        if model_name == "deep_mil":
            if torch is None or getattr(model, "model_", None) is None:
                raise RuntimeError("PyTorch is required to save deep_mil models.")
            torch.save(model.model_.state_dict(), model_path)
            deep_params = {
                "aa_dim": getattr(model, "aa_dim", None),
                "gene_dims": getattr(model, "gene_dims", None),
                "hidden_dim": getattr(model, "hidden_dim", None),
                "classifier_dim": getattr(model, "classifier_dim", None),
                "max_len": getattr(model, "max_len", None),
                "batch_size": getattr(model, "batch_size", None),
                "num_epochs": getattr(model, "num_epochs", None),
                "lr": getattr(model, "lr", None),
                "weight_decay": getattr(model, "weight_decay", None),
                "device": str(getattr(model, "device", "")),
            }
            with model_params_path.open("w") as f:
                json.dump(deep_params, f, indent=2, default=str)
            saved = True
        else:
            joblib.dump(model, model_path)
            saved = True

    if feature_info is not None:
        _save_feature_info(feature_info_path, feature_info)

    metrics["train_samples"] = len(metadata_df)
    metrics["model_path"] = str(model_path)
    if model_name == "deep_mil":
        metrics["model_params_path"] = str(model_params_path)
    metrics["feature_info_path"] = str(feature_info_path)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def train_all_datasets(
    train_root: str = config.TRAIN_ROOT,
    model_name: str = "kmer_logreg",
    feature_config: Dict[str, Any] = None,
    output_root: str = config.OUTPUT_DIR,
    cv_folds: int = 5,
    random_state: int = 123,
    test_root: str = config.TEST_ROOT,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train the specified model across all train_dataset_X directories.
    """
    feature_config = feature_config or {"use_kmers": True, "k": 3}
    dataset_map = data.list_datasets(train_root, test_root)

    summary: Dict[str, Any] = {}
    for dataset_name, info in dataset_map.items():
        res = train_on_dataset(
            dataset_name=dataset_name,
            train_path=info["train_path"],
            model_name=model_name,
            feature_config=feature_config,
            output_dir=output_root,
            cv_folds=cv_folds,
            random_state=random_state,
            model_params=model_params,
        )
        summary[dataset_name] = res.get("cv", {}).get("mean_auc")

    return summary
