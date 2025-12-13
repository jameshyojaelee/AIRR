"""
Sequence-level importance computation utilities.

Supports model-provided importance, optional SHAP projection for tabular models,
and simple dedup/near-duplicate collapsing to maximize coverage.
"""
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from airrml import config
from airrml.features import apply_kmer_features
from airrml.models.base import BaseRepertoireModel

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover
    shap = None


def _edit_distance_leq1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    # simple O(n) check for distance <= 1
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diff += 1
        if diff > 1:
            return False
        if len(a) > len(b):
            i += 1
        elif len(b) > len(a):
            j += 1
        else:
            i += 1
            j += 1
    diff += (len(a) - i) + (len(b) - j)
    return diff <= 1


def cluster_near_duplicates(df: pd.DataFrame, seq_col: str = "junction_aa", max_bucket_size: int = 5000) -> pd.DataFrame:
    """
    Collapse near-duplicate sequences (edit distance <=1). Keeps the highest scoring instance.
    """
    if df.empty:
        return df
    df = df.copy()
    df = df.sort_values("importance_score", ascending=False)
    buckets: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    kept_rows = []
    for _, row in df.iterrows():
        seq = str(row.get(seq_col, ""))
        if not seq:
            continue
        prefix = seq[:3]
        suffix = seq[-3:]
        candidates: list[str] = []
        for L in (len(seq) - 1, len(seq), len(seq) + 1):
            candidates.extend(buckets.get((L, prefix, suffix), []))
        if any(_edit_distance_leq1(seq, s) for s in candidates):
            continue
        kept_rows.append(row)
        key = (len(seq), prefix, suffix)
        buckets[key].append(seq)
        if len(buckets[key]) > max_bucket_size:
            buckets[key] = buckets[key][-max_bucket_size:]
    return pd.DataFrame(kept_rows)


def shap_score_sequences(
    model: BaseRepertoireModel,
    sequences_df: pd.DataFrame,
    feature_info: dict,
    sequence_col: str = "junction_aa",
) -> Optional[pd.DataFrame]:
    if shap is None:
        return None
    if "kmer" not in feature_info:
        return None
    X = apply_kmer_features(sequences_df, feature_info["kmer"], sequence_col=sequence_col)
    if X.empty:
        return None
    try:
        explainer = shap.Explainer(model.model_ if hasattr(model, "model_") else model)
        shap_vals = explainer(X)
        # Aggregate absolute shap values per sequence
        scores = np.abs(shap_vals.values).sum(axis=1)
        out = sequences_df.copy()
        out["importance_score"] = scores
        return out
    except Exception:
        return None


def score_sequences(
    model: BaseRepertoireModel,
    sequences_df: pd.DataFrame,
    feature_info: Optional[dict] = None,
    sequence_col: str = "junction_aa",
) -> pd.DataFrame:
    """
    Use a trained model to assign importance scores to sequences.
    Prefers model.get_sequence_importance; falls back to SHAP for tabular models.
    """
    if hasattr(model, "get_sequence_importance"):
        try:
            return model.get_sequence_importance(sequences_df, sequence_col=sequence_col)
        except NotImplementedError:
            pass
    if feature_info:
        shap_df = shap_score_sequences(model, sequences_df, feature_info, sequence_col=sequence_col)
        if shap_df is not None:
            return shap_df
    raise RuntimeError("Unable to score sequences; model does not support importance and SHAP failed.")


def select_top_sequences(
    scored_sequences: pd.DataFrame,
    top_k: int,
    score_col: str = "importance_score",
    dedup: bool = True,
) -> pd.DataFrame:
    """
    Select the top-k sequences by importance with optional near-duplicate collapse.
    """
    if scored_sequences.empty:
        return scored_sequences
    df = scored_sequences.dropna(subset=[score_col, "junction_aa"]).copy()
    df = df.sort_values(score_col, ascending=False)
    if dedup:
        df = cluster_near_duplicates(df)
    return df.head(top_k)


def format_sequence_rows(
    sequences_df: pd.DataFrame,
    dataset_name: str,
    start_id: int = 1,
    probability_placeholder: float = config.PROBABILITY_PLACEHOLDER,
) -> pd.DataFrame:
    """
    Format scored sequences into submission-ready rows.
    """
    df = sequences_df.reset_index(drop=True).copy()
    df.insert(0, "ID", [f"{dataset_name}_seq_top_{i}" for i in range(start_id, start_id + len(df))])
    df.insert(1, "dataset", dataset_name)
    df.insert(2, "label_positive_probability", probability_placeholder)
    for col in ["junction_aa", "v_call", "j_call"]:
        if col not in df.columns:
            df[col] = config.SEQUENCE_PLACEHOLDER
    return df[config.SUBMISSION_COLUMNS]
