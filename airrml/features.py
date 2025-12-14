"""
Feature engineering utilities for repertoire- and sequence-level models.

The functions here stay model-agnostic: they only depend on pandas/numpy and
can be reused across classical ML and deep learning pipelines.
"""
import hashlib
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from airrml import config


def _stable_hash(token: str, hash_size: int, hash_seed: int) -> int:
    digest = hashlib.md5(f"{hash_seed}:{token}".encode("utf-8")).hexdigest()
    return int(digest, 16) % hash_size


def _iter_kmers(seq: str, k: int) -> Iterable[str]:
    for i in range(len(seq) - k + 1):
        yield seq[i : i + k]


def _tfidf_transform(X: pd.DataFrame, idf: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    if X.empty:
        return X, idf if idf is not None else pd.Series(dtype=float)

    if idf is None:
        df = (X > 0).sum(axis=0)
        idf_vals = np.log((len(X) + 1) / (df + 1)) + 1.0
        idf = pd.Series(idf_vals, index=X.columns)
    tfidf = X.mul(idf, axis=1)
    norms = np.sqrt((tfidf**2).sum(axis=1))
    norms = norms.replace(0, 1.0)
    tfidf = tfidf.div(norms, axis=0)
    return tfidf, idf


def build_kmer_features(
    sequences_df: pd.DataFrame,
    k: Union[int, List[int]] = 3,
    sequence_col: str = "junction_aa",
    use_tfidf: bool = False,
    hashing: bool = False,
    hash_size: int = 2**15,
    hash_seed: int = 17,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute k-mer counts (optionally multi-k, TF-IDF weighted, and hashed) per repertoire ID.

    Returns a repertoire-level feature matrix X (index: ID) and feature_info describing
    the k-mer vocabulary/hash space and TF-IDF idf weights if enabled.
    """
    if "ID" not in sequences_df.columns:
        raise ValueError("sequences_df must contain an 'ID' column for repertoire grouping")

    k_list: List[int] = list(k) if isinstance(k, (list, tuple)) else [int(k)]
    valid = sequences_df.dropna(subset=["ID", sequence_col])
    rows = []
    prefix_cols = hashing or len(k_list) > 1

    for rep_id, seq_series in valid.groupby("ID")[sequence_col]:
        kmer_counts: Counter[str] = Counter()
        for seq in seq_series:
            if not isinstance(seq, str):
                continue
            for k_val in k_list:
                if len(seq) < k_val:
                    continue
                for kmer in _iter_kmers(seq, k_val):
                    token = f"{k_val}:{kmer}"
                    if hashing:
                        col = f"k{k_val}_h{_stable_hash(token, hash_size, hash_seed)}"
                    elif prefix_cols:
                        col = f"k{k_val}_{kmer}"
                    else:
                        col = kmer
                    kmer_counts[col] += 1
        row = {"ID": rep_id}
        row.update(kmer_counts)
        rows.append(row)

    if rows:
        X = pd.DataFrame(rows).set_index("ID").fillna(0)
        X = X.reindex(sorted(X.columns), axis=1)
    else:
        X = pd.DataFrame().set_index(pd.Index([], name="ID"))

    feature_info: Dict[str, Any] = {
        "type": "kmer",
        "k": k_list,
        "vocabulary": X.columns.tolist(),
        "hashing": hashing,
        "hash_size": hash_size,
        "hash_seed": hash_seed,
        "tfidf": use_tfidf,
    }

    if use_tfidf:
        X, idf = _tfidf_transform(X)
        feature_info["idf"] = idf.tolist()

    return X, feature_info


def apply_kmer_features(
    sequences_df: pd.DataFrame,
    feature_info: Dict[str, Any],
    sequence_col: str = "junction_aa",
) -> pd.DataFrame:
    """
    Apply a fitted k-mer vocabulary/hash space to new sequences to produce aligned features.
    """
    k = feature_info.get("k", 3)
    vocab = feature_info.get("vocabulary", [])
    use_tfidf = bool(feature_info.get("tfidf", False))
    hashing = bool(feature_info.get("hashing", False))
    hash_size = int(feature_info.get("hash_size", 2**15))
    hash_seed = int(feature_info.get("hash_seed", 17))

    X_raw, _ = build_kmer_features(
        sequences_df,
        k=k,
        sequence_col=sequence_col,
        use_tfidf=False,
        hashing=hashing,
        hash_size=hash_size,
        hash_seed=hash_seed,
    )
    X_aligned = X_raw.reindex(columns=vocab, fill_value=0)
    X_aligned = X_aligned.reindex(columns=vocab)

    if use_tfidf:
        idf_vals = feature_info.get("idf")
        idf = pd.Series(idf_vals, index=vocab) if idf_vals is not None else None
        X_aligned, _ = _tfidf_transform(X_aligned, idf=idf)

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


def build_tcrdist_nystrom_features(
    sequences_df: pd.DataFrame,
    dictionary_seqs: Optional[List[str]] = None,
    num_dict: int = 128,
    k: int = 3,
    sequence_col: str = "junction_aa",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Approximate TCRdist with Nyström-like features using k-mer cosine similarity to a dictionary of sequences.
    """
    if "ID" not in sequences_df.columns:
        raise ValueError("sequences_df must contain an 'ID' column for repertoire grouping")

    seqs = sequences_df.dropna(subset=[sequence_col])
    if dictionary_seqs is None:
        rng = np.random.default_rng(42)
        sampled = seqs[sequence_col].dropna().astype(str).unique()
        dictionary_seqs = rng.choice(sampled, size=min(num_dict, len(sampled)), replace=False).tolist() if len(sampled) else []

    dict_df = pd.DataFrame({"ID": [f"dict_{i}" for i in range(len(dictionary_seqs))], sequence_col: dictionary_seqs})
    dict_kmers, _ = build_kmer_features(dict_df, k=k, sequence_col=sequence_col, use_tfidf=False, hashing=False)

    # Build k-mer features for repertoires
    rep_kmers, _ = build_kmer_features(seqs, k=k, sequence_col=sequence_col, use_tfidf=False, hashing=False)
    rep_kmers = rep_kmers.reindex(columns=dict_kmers.columns, fill_value=0)
    dict_kmers = dict_kmers.reindex(columns=rep_kmers.columns, fill_value=0)

    # Cosine similarity to dictionary as features
    rep_mat = rep_kmers.to_numpy()
    dict_mat = dict_kmers.to_numpy()
    rep_norm = rep_mat / np.linalg.norm(rep_mat, axis=1, keepdims=True).clip(min=1e-9)
    dict_norm = dict_mat / np.linalg.norm(dict_mat, axis=1, keepdims=True).clip(min=1e-9)
    sim = rep_norm @ dict_norm.T  # [n_rep, num_dict]
    feat_cols = [f"tcrdist_dict_{i}" for i in range(sim.shape[1])]
    X = pd.DataFrame(sim, index=rep_kmers.index, columns=feat_cols)

    feature_info = {"vocabulary": feat_cols, "dictionary": dictionary_seqs, "k": k}
    return X, feature_info


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

def build_publicness_features(
    sequences_df: pd.DataFrame,
    existing_info: Optional[Dict[str, Any]] = None,
    sequence_col: str = "junction_aa",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute sequence publicness statistics (prevalence across repertoires).
    """
    if "ID" not in sequences_df.columns:
        raise ValueError("sequences_df must contain an 'ID' column for repertoire grouping")

    # In fit mode (existing_info=None): compute prevalence from current data
    if existing_info is None:
        valid = sequences_df.dropna(subset=["ID", sequence_col])
        # Count unique repertoires per sequence
        unique_pairs = valid[["ID", sequence_col]].drop_duplicates()
        counts = unique_pairs[sequence_col].value_counts()
        prevalence_map = counts.to_dict()
        feature_info = {"prevalence_map": prevalence_map}
    else:
        feature_info = existing_info
        prevalence_map = feature_info.get("prevalence_map", {})

    # Map prevalence to sequences
    # We work on a subset to save memory/time
    df = sequences_df[["ID", sequence_col]].copy()
    df[sequence_col] = df[sequence_col].fillna("").astype(str)
    # Default prevalence 0 if not in map (for new sequences in test time)
    # Optim: map is potentially large, use map or merge?
    # Merge is often faster for large DF
    prev_series = pd.Series(prevalence_map, name="prevalence")
    df = df.merge(prev_series, left_on=sequence_col, right_index=True, how="left")
    df["prevalence"] = df["prevalence"].fillna(0).astype(float)

    # Aggregate per repertoire
    # Features: mean, median, max, fraction > 1 (public)
    agg = df.groupby("ID")["prevalence"].agg(
        pub_mean="mean",
        pub_median="median",
        pub_max="max",
        pub_std="std",
        pub_prop=lambda x: (x > 1).mean()
    ).fillna(0)
    
    agg.index.name = "ID"
    # Ensure columns order
    cols = ["pub_mean", "pub_median", "pub_max", "pub_std", "pub_prop"]
    agg = agg.reindex(columns=cols, fill_value=0)
    
    return agg, feature_info


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
    use_tcrdist = config_dict.get("use_tcrdist_nystrom", False)
    use_publicness = config_dict.get("use_publicness", False)
    sequence_col = config_dict.get("sequence_col", config.SEQUENCE_COLS[0])
    if not any([use_kmers, use_vj, use_length, use_tcrdist, use_publicness]):
        raise ValueError("At least one feature type must be enabled")

    feature_info = feature_info or {}
    new_feature_info: Dict[str, Any] = dict(feature_info)
    feature_blocks = []

    if use_kmers:
        k = config_dict.get("k_list", config_dict.get("k", 3))
        use_tfidf = bool(config_dict.get("tfidf", False))
        hashing = bool(config_dict.get("hashing", False))
        hash_size = int(config_dict.get("hash_size", 2**15))
        hash_seed = int(config_dict.get("hash_seed", 17))

        if "kmer" in feature_info:
            X_kmer = apply_kmer_features(sequences_df, feature_info["kmer"])
        else:
            X_kmer, kmer_info = build_kmer_features(
                sequences_df,
                k=k,
                use_tfidf=use_tfidf,
                hashing=hashing,
                hash_size=hash_size,
                hash_seed=hash_seed,
            )
            new_feature_info["kmer"] = kmer_info
        feature_blocks.append(X_kmer)

    if use_tcrdist:
        if "tcrdist" in feature_info:
            # reuse stored dictionary
            dict_info = feature_info["tcrdist"]
            X_tcr, _ = build_tcrdist_nystrom_features(
                sequences_df,
                dictionary_seqs=dict_info.get("dictionary"),
                num_dict=len(dict_info.get("dictionary", [])),
                k=dict_info.get("k", 3),
                sequence_col=sequence_col,
            )
        else:
            X_tcr, tcr_info = build_tcrdist_nystrom_features(
                sequences_df,
                dictionary_seqs=None,
                num_dict=int(config_dict.get("tcrdist_num_dict", 128)),
                k=int(config_dict.get("tcrdist_k", 3)),
                sequence_col=sequence_col,
            )
            new_feature_info["tcrdist"] = tcr_info
        feature_blocks.append(X_tcr)

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

    if use_publicness:
        X_pub, pub_info = build_publicness_features(sequences_df, feature_info.get("publicness"), sequence_col=sequence_col)
        new_feature_info["publicness"] = pub_info
        feature_blocks.append(X_pub)

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
