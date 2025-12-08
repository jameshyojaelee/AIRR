"""
Feature engineering utilities for repertoire- and sequence-level models.

The functions here stay model-agnostic: they only depend on pandas/numpy and
can be reused across classical ML and deep learning pipelines.
"""
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from airrml import config


def build_kmer_features(
    sequences_df: pd.DataFrame,
    k: int = 3,
    sequence_col: str = "junction_aa",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute k-mer counts per repertoire ID.

    Returns a repertoire-level feature matrix X (index: ID) and feature_info
    describing the k-mer vocabulary.
    """
    if "ID" not in sequences_df.columns:
        raise ValueError("sequences_df must contain an 'ID' column for repertoire grouping")

    valid = sequences_df.dropna(subset=["ID", sequence_col])
    rows = []

    for rep_id, seq_series in valid.groupby("ID")[sequence_col]:
        kmer_counts: Counter[str] = Counter()
        for seq in seq_series:
            if not isinstance(seq, str):
                continue
            if len(seq) < k:
                continue
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                kmer_counts[kmer] += 1
        row = {"ID": rep_id}
        row.update(kmer_counts)
        rows.append(row)

    if rows:
        X = pd.DataFrame(rows).set_index("ID").fillna(0)
        X = X.reindex(sorted(X.columns), axis=1)
    else:
        X = pd.DataFrame().set_index(pd.Index([], name="ID"))

    feature_info = {"type": "kmer", "k": k, "vocabulary": X.columns.tolist()}
    return X, feature_info


def apply_kmer_features(
    sequences_df: pd.DataFrame,
    feature_info: Dict[str, Any],
    sequence_col: str = "junction_aa",
) -> pd.DataFrame:
    """
    Apply a fitted k-mer vocabulary to new sequences to produce aligned features.
    """
    k = int(feature_info.get("k", 3))
    vocab = feature_info.get("vocabulary", [])

    X_raw, _ = build_kmer_features(sequences_df, k=k, sequence_col=sequence_col)
    X_aligned = X_raw.reindex(columns=vocab, fill_value=0)
    X_aligned = X_aligned.reindex(columns=vocab)
    return X_aligned


def build_vj_usage_features(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate V and J gene usage per repertoire ID.
    """
    if "ID" not in sequences_df.columns:
        raise ValueError("sequences_df must contain an 'ID' column for repertoire grouping")

    valid = sequences_df.dropna(subset=["ID"])

    v_dummies = pd.get_dummies(valid["v_call"], prefix="v")
    j_dummies = pd.get_dummies(valid["j_call"], prefix="j")

    v_counts = v_dummies.groupby(valid["ID"]).sum()
    j_counts = j_dummies.groupby(valid["ID"]).sum()

    X = pd.concat([v_counts, j_counts], axis=1).fillna(0)
    X.index.name = "ID"
    if not X.empty:
        X = X.reindex(sorted(X.columns), axis=1)
    return X


def build_length_features(
    sequences_df: pd.DataFrame,
    existing_info: Optional[Dict[str, Any]] = None,
    sequence_col: str = "junction_aa",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute simple CDR3 length statistics per repertoire.
    """
    if "ID" not in sequences_df.columns:
        raise ValueError("sequences_df must contain an 'ID' column for repertoire grouping")

    valid = sequences_df.dropna(subset=["ID", sequence_col]).copy()
    valid["seq_len"] = valid[sequence_col].astype(str).str.len()

    agg = valid.groupby("ID")["seq_len"].agg(["mean", "std", "min", "max", "median"]).fillna(0)
    agg.index.name = "ID"
    col_order = ["len_mean", "len_std", "len_min", "len_max", "len_median"]
    agg.columns = col_order
    feature_info = existing_info or {"vocabulary": col_order}
    agg = agg.reindex(columns=feature_info["vocabulary"], fill_value=0)
    return agg, feature_info


def apply_vj_features(sequences_df: pd.DataFrame, feature_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a fixed V/J vocabulary to new data.
    """
    vocab = feature_info.get("vocabulary", [])
    X_raw = build_vj_usage_features(sequences_df)
    X_aligned = X_raw.reindex(columns=vocab, fill_value=0)
    X_aligned = X_aligned.reindex(columns=vocab)
    return X_aligned


def build_combined_feature_matrix(
    sequences_df: pd.DataFrame,
    label_df: Optional[pd.DataFrame],
    config_dict: Dict[str, Any],
    feature_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Config-driven feature builder that can operate in fit (feature_info=None)
    or transform (feature_info provided) mode.
    """
    use_kmers = config_dict.get("use_kmers", True)
    use_vj = config_dict.get("use_vj", False)
    use_length = config_dict.get("use_length", False)
    if not use_kmers and not use_vj and not use_length:
        raise ValueError("At least one feature type must be enabled (k-mers, VJ usage, or length stats)")

    feature_info = feature_info or {}
    new_feature_info: Dict[str, Any] = dict(feature_info)
    feature_blocks = []

    if use_kmers:
        k = int(config_dict.get("k", 3))
        if "kmer" in feature_info:
            X_kmer = apply_kmer_features(sequences_df, feature_info["kmer"])
        else:
            X_kmer, kmer_info = build_kmer_features(sequences_df, k=k)
            new_feature_info["kmer"] = kmer_info
        feature_blocks.append(X_kmer)

    if use_vj:
        if "vj" in feature_info:
            X_vj = apply_vj_features(sequences_df, feature_info["vj"])
        else:
            X_vj = build_vj_usage_features(sequences_df)
            new_feature_info["vj"] = {"vocabulary": X_vj.columns.tolist()}
        feature_blocks.append(X_vj)

    if use_length:
        X_len, len_info = build_length_features(sequences_df, feature_info.get("length"))
        new_feature_info["length"] = len_info
        feature_blocks.append(X_len)

    X = pd.concat(feature_blocks, axis=1) if feature_blocks else pd.DataFrame()

    y: Optional[pd.Series] = None
    if label_df is not None:
        labels = label_df.copy()
        if "ID" in labels.columns:
            labels = labels.set_index("ID")
        if config.LABEL_COL in labels.columns and not X.empty:
            common_ids = X.index.intersection(labels.index)
            X = X.loc[common_ids]
            y = labels.loc[common_ids, config.LABEL_COL]
        elif config.LABEL_COL in labels.columns:
            y = labels[config.LABEL_COL]

    return X, y, new_feature_info
