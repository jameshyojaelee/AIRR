"""
Evaluation helpers for repertoire-level metrics, sequence-level set metrics,
and cross-validation workflows.
"""
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from airrml import config
from airrml.features import build_combined_feature_matrix
from airrml.models import get_model


def compute_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute ROC AUC for repertoire-level predictions with basic validation.
    """
    if len(y_true) == 0:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        # Handle cases with a single class in y_true
        return float("nan")


def compute_jaccard(submitted_seqs: Iterable[str], true_seqs: Iterable[str]) -> float:
    """
    Compute Jaccard similarity for label-associated sequence sets.
    """
    submitted_set = set(submitted_seqs)
    true_set = set(true_seqs)
    if not submitted_set and not true_set:
        return 1.0
    if not submitted_set or not true_set:
        return 0.0
    return len(submitted_set & true_set) / len(submitted_set | true_set)


def evaluate_dataset(
    predictions_df: pd.DataFrame,
    important_seqs_df: pd.DataFrame,
    ground_truth_labels: pd.DataFrame,
    ground_truth_seqs: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate a single dataset on both competition objectives.
    """
    raise NotImplementedError("Full evaluation requires ground-truth access and is left for offline validation.")


def cross_validate_model(
    model_name: str,
    sequences_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_config: Dict[str, Any],
    cv_folds: int = 5,
    random_state: int = 123,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run repertoire-level cross-validation within a single dataset.
    """
    results: Dict[str, Any] = {"fold_scores": [], "fold_top_sequences": []}

    if cv_folds is None or cv_folds < 2:
        results["mean_auc"] = float("nan")
        results["std_auc"] = float("nan")
        return results

    # Prepare label series indexed by ID
    labels = label_df.copy()
    if "ID" in labels.columns:
        labels = labels.set_index("ID")
    if config.LABEL_COL not in labels.columns:
        raise ValueError(f"label_df must contain column '{config.LABEL_COL}'")
    y_full = labels[config.LABEL_COL].astype(int)

    # Handle sequence-consuming models separately (operate on sequences directly)
    probe_model = get_model(model_name, random_state=random_state, **(model_params or {}))
    if getattr(probe_model, "consumes_sequences", False):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(np.zeros(len(y_full)), y_full):
            train_ids = y_full.index[train_idx]
            val_ids = y_full.index[val_idx]
            train_seq = sequences_df[sequences_df["ID"].isin(train_ids)]
            val_seq = sequences_df[sequences_df["ID"].isin(val_ids)]
            train_labels = y_full.loc[train_ids]
            val_labels = y_full.loc[val_ids]

            model = get_model(model_name, random_state=random_state, **(model_params or {}))
            model.fit(train_seq, train_labels)
            raw_probs = model.predict_proba(val_seq)
            # Align predictions to validation label order to avoid AUC misalignment
            prob_series = pd.Series(raw_probs, index=val_seq["ID"].drop_duplicates())
            val_probs = prob_series.reindex(val_ids).to_numpy()
            auc = compute_auc(val_labels.to_numpy(), val_probs)
            results["fold_scores"].append(auc)

            # Task 2 stability check: extract top sequences
            try:
                imp = model.get_sequence_importance(train_seq)
                if imp is not None and not imp.empty:
                    # Sort by score descending
                    if "importance_score" in imp.columns:
                        imp = imp.sort_values("importance_score", ascending=False)
                    top_seqs = imp.head(2000)
                    seq_keys = set((top_seqs["junction_aa"].astype(str) + "|" + top_seqs["v_call"].astype(str) + "|" + top_seqs["j_call"].astype(str)).tolist())
                    results["fold_top_sequences"].append(seq_keys)
            except Exception as e:
                print(f"Warning: failed to extract sequence importance for fold: {e}")
        scores = np.array(results["fold_scores"])
        results["mean_auc"] = float(np.nanmean(scores)) if len(scores) else float("nan")
        results["std_auc"] = float(np.nanstd(scores)) if len(scores) else float("nan")

        # Compute stability (mean pairwise Jaccard)
        seq_sets = results.get("fold_top_sequences", [])
        if len(seq_sets) > 1:
            pair_scores = []
            for a, b in combinations(seq_sets, 2):
                pair_scores.append(compute_jaccard(list(a), list(b)))
            results["mean_stability"] = float(np.mean(pair_scores))
        else:
            results["mean_stability"] = float("nan")

        return results

    # Classical models: build features per-fold to avoid leakage
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(np.zeros(len(y_full)), y_full):
        train_ids = y_full.index[train_idx]
        val_ids = y_full.index[val_idx]

        train_seq = sequences_df[sequences_df["ID"].isin(train_ids)]
        val_seq = sequences_df[sequences_df["ID"].isin(val_ids)]
        train_label_df = label_df[label_df["ID"].isin(train_ids)]
        val_label_df = label_df[label_df["ID"].isin(val_ids)]

        X_train, y_train, feature_info = build_combined_feature_matrix(train_seq, train_label_df, feature_config)
        X_val, y_val, _ = build_combined_feature_matrix(val_seq, val_label_df, feature_config, feature_info=feature_info)

        model = get_model(model_name, random_state=random_state, **(model_params or {}))
        model.set_feature_info(feature_info)
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)
        auc = compute_auc(y_val.to_numpy(), val_probs)
        results["fold_scores"].append(auc)

        # Task 2 stability check: extract top sequences
        try:
            # Reconstruct sequence dataframe from X_train if possible?
            # Classical models (kmer_logreg) can project back to sequences if they implement get_sequence_importance
            # We pass the original `train_seq` for that
            imp = model.get_sequence_importance(train_seq)
            if imp is not None and not imp.empty:
                if "importance_score" in imp.columns:
                    imp = imp.sort_values("importance_score", ascending=False)
                top_seqs = imp.head(2000)
                seq_keys = set((top_seqs["junction_aa"].astype(str) + "|" + top_seqs["v_call"].astype(str) + "|" + top_seqs["j_call"].astype(str)).tolist())
                results["fold_top_sequences"].append(seq_keys)
        except Exception:
            pass

    scores = np.array(results["fold_scores"])
    results["mean_auc"] = float(np.nanmean(scores)) if len(scores) else float("nan")
    results["std_auc"] = float(np.nanstd(scores)) if len(scores) else float("nan")

    # Compute stability (mean pairwise Jaccard)
    seq_sets = results.get("fold_top_sequences", [])
    if len(seq_sets) > 1:
        pair_scores = []
        for a, b in combinations(seq_sets, 2):
            pair_scores.append(compute_jaccard(list(a), list(b)))
        results["mean_stability"] = float(np.mean(pair_scores))
    else:
        results["mean_stability"] = float("nan")

    return results
