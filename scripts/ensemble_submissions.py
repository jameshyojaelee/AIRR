#!/usr/bin/env python3
"""Ensemble multiple AIRR-ML submission CSVs.

This script combines:
- Task 1: ensemble repertoire probabilities (weighted average).
- Task 2: ensemble ranked sequence lists (rank aggregation; default = RRF).

The output is a schema-correct `submission.csv` that passes `scripts/check_submission.py`.

Notes:
- Sequence rows are detected by non-empty `junction_aa`.
- Ranks are recovered from `ID` via the `_seq_top_{rank}` suffix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airrml import config  # noqa: E402
from airrml import seq_importance  # noqa: E402


def _is_blank_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).eq("")


def _read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    cols = list(df.columns)
    if cols != config.SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns mismatch for {path}. Got {cols}, expected {config.SUBMISSION_COLUMNS}")
    return df


_RANK_RE = re.compile(r"_seq_top_(\d+)$")


def _extract_rank(id_series: pd.Series) -> pd.Series:
    ranks = id_series.astype(str).str.extract(_RANK_RE, expand=False)
    return pd.to_numeric(ranks, errors="coerce")


def _normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float) / float(n)
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if float(w.sum()) == 0.0:
        raise ValueError("sum(weights) must be > 0")
    return w / float(w.sum())


def _infer_train_dataset_from_test_name(test_dataset: str) -> Optional[str]:
    m = re.match(r"^test_dataset_(\d+)", str(test_dataset))
    if not m:
        return None
    return f"train_dataset_{m.group(1)}"


@dataclass(frozen=True)
class DatasetStrategy:
    task1_sources: List[int]
    task1_weights: np.ndarray
    task2_sources: List[int]
    task2_weights: np.ndarray


def _load_strategy(path: Path, n_sources: int) -> Dict[str, DatasetStrategy]:
    """Load a per-train-dataset strategy JSON.

    Expected format:
    {
      "task1": {
        "train_dataset_1": {"sources": [0,2], "weights": [0.7,0.3]},
        ...
      },
      "task2": {
        "train_dataset_1": {"sources": [1,2], "weights": [1.0,1.0]},
        ...
      }
    }
    Where sources refer to indices in --submissions.
    """
    raw = json.loads(path.read_text())
    task1 = raw.get("task1", {}) or {}
    task2 = raw.get("task2", {}) or {}
    out: Dict[str, DatasetStrategy] = {}
    all_keys = set(task1.keys()) | set(task2.keys())
    for ds in all_keys:
        t1 = task1.get(ds, {}) or {}
        t2 = task2.get(ds, {}) or {}
        t1_sources = list(map(int, t1.get("sources", list(range(n_sources)))))
        t2_sources = list(map(int, t2.get("sources", list(range(n_sources)))))
        if any(i < 0 or i >= n_sources for i in t1_sources + t2_sources):
            raise ValueError(f"Invalid source index in strategy for {ds}")
        t1_w = _normalize_weights(t1.get("weights"), len(t1_sources))
        t2_w = _normalize_weights(t2.get("weights"), len(t2_sources))
        out[ds] = DatasetStrategy(task1_sources=t1_sources, task1_weights=t1_w, task2_sources=t2_sources, task2_weights=t2_w)
    return out


def ensemble_task1(
    submissions: List[pd.DataFrame],
    weights: np.ndarray,
    strategy: Optional[Dict[str, DatasetStrategy]] = None,
) -> pd.DataFrame:
    """Ensemble repertoire probabilities across submissions (Task 1)."""
    # Gather per-submission series of probabilities
    prob_series: List[pd.Series] = []
    all_index: Optional[pd.MultiIndex] = None
    for df in submissions:
        is_rep = _is_blank_series(df["junction_aa"])
        rep = df[is_rep].copy()
        s = rep.set_index(["dataset", "ID"])["label_positive_probability"].astype(float)
        prob_series.append(s)
        all_index = s.index if all_index is None else all_index.union(s.index)

    if all_index is None or len(all_index) == 0:
        raise ValueError("No repertoire rows found in any submission.")

    weighted_sum = pd.Series(0.0, index=all_index)
    weight_sum = pd.Series(0.0, index=all_index)

    # Apply either global weights or per-train-dataset strategy
    if strategy is None:
        for w, s in zip(weights, prob_series):
            weighted_sum.loc[s.index] += w * s
            weight_sum.loc[s.index] += w
    else:
        # Group test datasets by inferred train dataset
        index_df = all_index.to_frame(index=False)
        index_df["train_dataset"] = index_df["dataset"].map(_infer_train_dataset_from_test_name)
        for train_ds, rows in index_df.groupby("train_dataset", dropna=False, observed=True):
            idx = pd.MultiIndex.from_frame(rows[["dataset", "ID"]])
            if train_ds is None or train_ds not in strategy:
                # Fall back to global weights
                for w, s in zip(weights, prob_series):
                    common = s.index.intersection(idx)
                    if len(common) == 0:
                        continue
                    weighted_sum.loc[common] += w * s.loc[common]
                    weight_sum.loc[common] += w
                continue
            strat = strategy[train_ds]
            for src_i, w in zip(strat.task1_sources, strat.task1_weights):
                s = prob_series[src_i]
                common = s.index.intersection(idx)
                if len(common) == 0:
                    continue
                weighted_sum.loc[common] += w * s.loc[common]
                weight_sum.loc[common] += w

    missing = weight_sum[weight_sum <= 0].index
    if len(missing):
        raise ValueError(f"Missing Task 1 predictions for {len(missing)} (dataset,ID) pairs after ensembling.")

    probs = (weighted_sum / weight_sum).clip(0.0, 1.0)
    out = probs.reset_index()
    out = out.rename(columns={0: "label_positive_probability"})
    out["label_positive_probability"] = probs.values
    out["junction_aa"] = config.SEQUENCE_PLACEHOLDER
    out["v_call"] = config.SEQUENCE_PLACEHOLDER
    out["j_call"] = config.SEQUENCE_PLACEHOLDER
    return out[config.SUBMISSION_COLUMNS]


def _rrf_score(rank: pd.Series, rrf_k: int) -> pd.Series:
    # rank: 1..K
    return 1.0 / (float(rrf_k) + rank.astype(float).clip(lower=1.0))


def _borda_score(rank: pd.Series, max_rank: pd.Series) -> pd.Series:
    # Higher rank (closer to 1) => higher score in [0,1]
    denom = (max_rank.astype(float) + 1.0).clip(lower=2.0)
    return 1.0 - (rank.astype(float) / denom)


def ensemble_task2(
    submissions: List[pd.DataFrame],
    weights: np.ndarray,
    top_k: int,
    method: str = "rrf",
    rrf_k: int = 60,
    dedup_near: bool = False,
    candidate_multiplier: int = 5,
    strategy: Optional[Dict[str, DatasetStrategy]] = None,
) -> pd.DataFrame:
    """Ensemble ranked sequence lists across submissions (Task 2)."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be >= 1")

    per_source_rows: List[pd.DataFrame] = []
    for df in submissions:
        is_seq = ~_is_blank_series(df["junction_aa"])
        seq = df[is_seq].copy()
        # Keep only training dataset names for Task 2
        seq = seq[seq["dataset"].astype(str).str.startswith("train_dataset_")].copy()
        if seq.empty:
            per_source_rows.append(pd.DataFrame(columns=["dataset", "junction_aa", "v_call", "j_call", "rank"]))
            continue
        seq["rank"] = _extract_rank(seq["ID"])
        seq = seq.dropna(subset=["rank"]).copy()
        seq["rank"] = seq["rank"].astype(int)
        # guard against malformed negative ranks
        seq = seq[seq["rank"] > 0].copy()
        per_source_rows.append(seq[["dataset", "junction_aa", "v_call", "j_call", "rank"]])

    # Determine which training datasets exist across any source
    train_datasets: List[str] = sorted(
        set().union(*[set(df["dataset"].unique().tolist()) for df in per_source_rows if not df.empty])
    )
    if not train_datasets:
        raise ValueError("No Task 2 sequence rows found in any submission.")

    all_out_rows: List[pd.DataFrame] = []
    for train_ds in train_datasets:
        if strategy is None or train_ds not in strategy:
            src_ids = list(range(len(submissions)))
            src_w = weights
        else:
            strat = strategy[train_ds]
            src_ids = strat.task2_sources
            src_w = strat.task2_weights

        # Compute aggregated score per (junction_aa,v_call,j_call)
        pieces: List[pd.DataFrame] = []
        for src_i, w in zip(src_ids, src_w):
            seq = per_source_rows[src_i]
            seq = seq[seq["dataset"] == train_ds].copy()
            if seq.empty:
                continue
            if method == "rrf":
                seq["score"] = _rrf_score(seq["rank"], rrf_k=rrf_k) * float(w)
            elif method == "borda":
                max_rank = seq["rank"].max()
                seq["score"] = _borda_score(seq["rank"], max_rank=pd.Series(max_rank, index=seq.index)) * float(w)
            else:
                raise ValueError(f"Unsupported method: {method}")
            pieces.append(seq[["junction_aa", "v_call", "j_call", "score"]])

        if not pieces:
            continue
        merged = pd.concat(pieces, ignore_index=True)
        merged["junction_aa"] = merged["junction_aa"].astype(str)
        merged["v_call"] = merged["v_call"].astype(str)
        merged["j_call"] = merged["j_call"].astype(str)
        merged = merged[merged["junction_aa"].astype(str).str.len() > 0].copy()

        agg = merged.groupby(["junction_aa", "v_call", "j_call"], observed=True)["score"].sum().reset_index()
        agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

        # Optional near-duplicate collapse. Only consider a limited candidate set for speed.
        if dedup_near:
            cand = agg.head(top_k * candidate_multiplier).rename(columns={"score": "importance_score"})
            cand = seq_importance.cluster_near_duplicates(cand, seq_col="junction_aa")
            cand = cand.rename(columns={"importance_score": "score"})
            agg = cand.sort_values("score", ascending=False).reset_index(drop=True)

        top = agg.head(top_k).copy()
        top["dataset"] = train_ds
        top["label_positive_probability"] = float(config.PROBABILITY_PLACEHOLDER)
        top["ID"] = [f"{train_ds}_seq_top_{i}" for i in range(1, len(top) + 1)]
        all_out_rows.append(top[config.SUBMISSION_COLUMNS])

    if not all_out_rows:
        raise ValueError("No Task 2 rows were generated.")
    return pd.concat(all_out_rows, ignore_index=True)[config.SUBMISSION_COLUMNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble AIRR-ML submissions (Task1+Task2).")
    p.add_argument("--submissions", nargs="+", type=Path, required=True, help="List of submission CSV paths")
    p.add_argument("--weights-task1", nargs="*", type=float, default=None, help="Weights for Task 1 ensembling (same length as submissions)")
    p.add_argument("--weights-task2", nargs="*", type=float, default=None, help="Weights for Task 2 ensembling (same length as submissions)")
    p.add_argument("--task2-method", choices=["rrf", "borda"], default="rrf")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--top-k", type=int, default=50000, help="Top-K sequences per training dataset for Task 2")
    p.add_argument("--dedup-near", action="store_true", help="Collapse near-duplicate sequences (edit distance <=1)")
    p.add_argument("--candidate-multiplier", type=int, default=5, help="Candidates = top_k * multiplier for near-dedup")
    p.add_argument("--strategy", type=Path, default=None, help="Optional per-train-dataset mixing strategy JSON")
    p.add_argument("--output", type=Path, required=True, help="Output submission CSV path")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    submissions = []
    for path in args.submissions:
        if not path.exists():
            raise FileNotFoundError(f"submission not found: {path}")
        submissions.append(_read_submission(path))

    w1 = _normalize_weights(args.weights_task1 if args.weights_task1 else None, len(submissions))
    w2 = _normalize_weights(args.weights_task2 if args.weights_task2 else None, len(submissions))

    strategy = _load_strategy(args.strategy, len(submissions)) if args.strategy else None

    rep = ensemble_task1(submissions, weights=w1, strategy=strategy)
    seq = ensemble_task2(
        submissions,
        weights=w2,
        top_k=int(args.top_k),
        method=args.task2_method,
        rrf_k=int(args.rrf_k),
        dedup_near=bool(args.dedup_near),
        candidate_multiplier=int(args.candidate_multiplier),
        strategy=strategy,
    )

    out = pd.concat([seq, rep], ignore_index=True)[config.SUBMISSION_COLUMNS]
    out["ID"] = out["ID"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out["junction_aa"] = out["junction_aa"].astype(str)
    out["v_call"] = out["v_call"].astype(str)
    out["j_call"] = out["j_call"].astype(str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote ensembled submission: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

