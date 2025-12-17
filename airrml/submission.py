"""
Submission building utilities for AIRR-ML-25.

Constructs repertoire-level prediction rows and sequence-level label-associated
rows, then merges them into a single submission.csv following the competition
schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import pandas as pd
import warnings

from airrml import config, data
from airrml.features import build_combined_feature_matrix
from airrml.models import get_model
from airrml import seq_importance

try:  # Optional torch dependency for deep models
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_model_artifacts(model_dir: Path, model_name: str):
    model_joblib = model_dir / "model.joblib"
    model_pt = model_dir / "model.pt"
    model_params_path = model_dir / "model_params.json"
    feature_info_path = model_dir / "feature_info.json"

    model = None
    feature_info = None

    if feature_info_path.exists():
        with feature_info_path.open("r") as f:
            feature_info = json.load(f)

    if model_joblib.exists():
        model = joblib.load(model_joblib)
    elif model_pt.exists():
        if torch is None:
            raise ImportError("PyTorch is required to load deep models saved as .pt")
        params: Dict[str, Any] = {}
        if model_params_path.exists():
            with model_params_path.open("r") as f:
                params = json.load(f)
        # Fill missing deep hyperparams from feature_info (preferred, since it's written after training).
        if feature_info and isinstance(feature_info, dict):
            for key in ("model_dim", "num_heads", "num_layers", "ff_dim", "dropout", "max_len", "max_sequences_per_rep"):
                if key in feature_info and (key not in params or params.get(key) is None):
                    params[key] = feature_info[key]

        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Prefer constructing the model directly on the target device for speed.
        params = dict(params)
        params.setdefault("device", str(target_device))
        model = get_model(model_name, **params)
        model.device = target_device  # type: ignore[attr-defined]

        # Rebuild network using saved vocab sizes if available
        if feature_info and isinstance(feature_info, dict):
            gene_vocabs = feature_info.get("gene_vocabs", {}) or {}
            if hasattr(model, "_build_gene_vocabs") and gene_vocabs:
                model.gene_vocabs_ = gene_vocabs
            if "max_len" in feature_info and hasattr(model, "max_len"):
                model.max_len = feature_info["max_len"]

        if getattr(model, "model_", None) is None:
            from airrml.models.deep_mil import DeepMILNet

            gene_vocab_sizes = {k: len(v) for k, v in getattr(model, "gene_vocabs_", {}).items()}
            model_dim = int(getattr(model, "model_dim", params.get("model_dim", 256)))
            num_heads = int(getattr(model, "num_heads", params.get("num_heads", 8)))
            num_layers = int(getattr(model, "num_layers", params.get("num_layers", 4)))
            ff_dim = int(getattr(model, "ff_dim", params.get("ff_dim", model_dim * 4)))
            dropout = float(getattr(model, "dropout", params.get("dropout", 0.1)))
            max_len = int(getattr(model, "max_len", params.get("max_len", 60)))
            gene_dims = params.get("gene_dims", getattr(model, "gene_dims", {"v": 16, "j": 8}))
            classifier_dim = int(params.get("classifier_dim", getattr(model, "classifier_dim", 128)))

            model.model_ = DeepMILNet(
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                dropout=dropout,
                max_len=max_len,
                gene_vocab_sizes=gene_vocab_sizes,
                gene_dims=gene_dims,
                classifier_dim=classifier_dim,
            ).to(target_device)
        state = torch.load(model_pt, map_location=target_device)
        if getattr(model, "model_", None) is None:
            raise RuntimeError("Deep model architecture was not initialized before loading state dict.")
        model.model_.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Could not find model artifacts in {model_dir}")

    return model, feature_info


def build_repertoire_predictions(
    model,
    feature_info: Dict[str, Any],
    test_sequences_df: pd.DataFrame,
    test_repertoires_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Build repertoire-level prediction rows for a single test dataset.
    """
    # Sequence-consuming models consume sequences directly
    if getattr(model, "consumes_sequences", False):
        probs = model.predict_proba(test_sequences_df)
    else:
        feature_info = feature_info or {}
        # Build the same feature blocks used during training based on persisted feature_info.
        # This avoids silently dropping feature blocks (e.g., length/publicness/tcrdist) at inference time.
        config_dict = {
            "use_kmers": "kmer" in feature_info,
            "use_vj": "vj" in feature_info,
            "use_length": "length" in feature_info,
            "use_tcrdist_nystrom": "tcrdist" in feature_info,
            "use_publicness": "publicness" in feature_info,
        }
        if not any(config_dict.values()):
            # Fallback for legacy artifacts that may not persist feature_info blocks.
            config_dict = {"use_kmers": True}
        X_test, _, _ = build_combined_feature_matrix(test_sequences_df, test_repertoires_df, config_dict, feature_info=feature_info)  # type: ignore
        if hasattr(model, "set_feature_info"):
            model.set_feature_info(feature_info)
        probs = model.predict_proba(X_test)

    ids = list(test_repertoires_df["ID"]) if "ID" in test_repertoires_df.columns else X_test.index.tolist()
    df = pd.DataFrame(
        {
            "ID": ids,
            "dataset": dataset_name,
            "label_positive_probability": probs,
            "junction_aa": config.SEQUENCE_PLACEHOLDER,
            "v_call": config.SEQUENCE_PLACEHOLDER,
            "j_call": config.SEQUENCE_PLACEHOLDER,
        }
    )
    return df[config.SUBMISSION_COLUMNS]


def build_sequence_importance(
    model,
    train_sequences_df: pd.DataFrame,
    dataset_name: str,
    top_k: int = 50000,
    feature_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build sequence-level rows using model-provided importance scores.
    """
    # Fast path for models that precompute a full ranked list during fit (e.g., enrichment).
    precomputed = getattr(model, "sequence_stats_", None)
    if isinstance(precomputed, pd.DataFrame) and not precomputed.empty and "importance_score" in precomputed.columns:
        cols = ["junction_aa", "v_call", "j_call", "importance_score"]
        if all(c in precomputed.columns for c in cols):
            candidates = precomputed[cols].copy()
            candidates = candidates.sort_values("importance_score", ascending=False)
            candidates = candidates.head(top_k)
            # Near-duplicate collapsing is optional; keep exact top-k for speed.
            candidates = seq_importance.select_top_sequences(candidates, top_k=top_k, dedup=False)
            return seq_importance.format_sequence_rows(candidates, dataset_name)

    # Deduplicate sequences and cap to avoid OOM
    seq_cols = ["junction_aa", "v_call", "j_call"]
    unique_seqs = train_sequences_df[seq_cols].drop_duplicates()
    is_deep = hasattr(model, "model_") and model.__class__.__name__.lower().startswith("deep")
    max_seqs = top_k if is_deep else max(top_k * 5, top_k)
    if len(unique_seqs) > max_seqs:
        unique_seqs = unique_seqs.head(max_seqs)

    scored = seq_importance.score_sequences(model, unique_seqs, feature_info=feature_info)
    if scored is None or scored.empty:
        raise RuntimeError("Model did not return sequence importance scores.")

    scored = seq_importance.select_top_sequences(scored, top_k=top_k)

    return seq_importance.format_sequence_rows(scored, dataset_name)


def assemble_submission(
    train_root: str,
    test_root: str,
    model_name: str,
    model_output_root: str,
    top_k_sequences: int,
    submission_path: str,
) -> None:
    """
    Build a full submission.csv by running inference over all test datasets and
    extracting sequence-level importance from training datasets.
    """
    train_root = Path(train_root)
    test_root = Path(test_root)
    model_output_root = Path(model_output_root)
    submission_path = Path(submission_path)

    all_rows: List[pd.DataFrame] = []

    dataset_map = data.list_datasets(train_root, test_root)
    for dataset_name, info in dataset_map.items():
        model_dir = model_output_root / dataset_name
        model, feature_info = _load_model_artifacts(model_dir, model_name)

        # Load train data only if needed for Task 2 importance.
        # Some models (e.g., enrichment) precompute a ranked list during fit.
        needs_train_for_importance = True
        precomputed = getattr(model, "sequence_stats_", None)
        if isinstance(precomputed, pd.DataFrame) and not precomputed.empty and "importance_score" in precomputed.columns:
            cols = ["junction_aa", "v_call", "j_call", "importance_score"]
            if all(c in precomputed.columns for c in cols):
                needs_train_for_importance = False

        train_seq_df: pd.DataFrame
        if needs_train_for_importance:
            train_seq_df, _ = data.load_full_dataset(info["train_path"])
        else:
            train_seq_df = pd.DataFrame(columns=["junction_aa", "v_call", "j_call"])

        # Sequence importance (Task 2)
        try:
            seq_rows = build_sequence_importance(model, train_seq_df, dataset_name, top_k=top_k_sequences, feature_info=feature_info)
            all_rows.append(seq_rows)
        except Exception as exc:
            warnings.warn(f"Skipping sequence importance for {dataset_name}: {exc}")

        # Repertoire predictions for each associated test dataset
        for test_path in info["test_paths"]:
            test_seq_df, test_meta_df = data.load_test_dataset(test_path)
            preds = build_repertoire_predictions(model, feature_info, test_seq_df, test_meta_df, dataset_name=test_path.name)
            all_rows.append(preds)

    if not all_rows:
        raise RuntimeError("No submission rows were generated.")

    submission_df = pd.concat(all_rows, ignore_index=True)
    submission_df = submission_df[config.SUBMISSION_COLUMNS]

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)


__all__ = [
    "build_repertoire_predictions",
    "build_sequence_importance",
    "assemble_submission",
]
