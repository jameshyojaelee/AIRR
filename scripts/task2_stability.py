#!/usr/bin/env python3
"""Compute Task-2 stability (proxy) across multiple submissions.

We don't have ground-truth Task-2 labels locally, so this computes a proxy:
Jaccard overlap of the top-N submitted sequences across multiple runs.

Usage:
  python3 scripts/task2_stability.py --submissions outputs/*/submission.csv --top-n 10000
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airrml import config  # noqa: E402


def _is_blank_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).eq("")


_RANK_RE = re.compile(r"_seq_top_(\d+)$")


def _extract_rank(id_series: pd.Series) -> pd.Series:
    ranks = id_series.astype(str).str.extract(_RANK_RE, expand=False)
    return pd.to_numeric(ranks, errors="coerce")


def _seq_key(df: pd.DataFrame) -> pd.Series:
    return df["junction_aa"].astype(str) + "|" + df["v_call"].astype(str) + "|" + df["j_call"].astype(str)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter / uni) if uni else 0.0


def load_topn_sets(submission_path: Path, top_n: int) -> Dict[str, Set[str]]:
    df = pd.read_csv(submission_path, keep_default_na=False)
    cols = list(df.columns)
    if cols != config.SUBMISSION_COLUMNS:
        raise ValueError(f"columns mismatch for {submission_path}. Got {cols}, expected {config.SUBMISSION_COLUMNS}")

    is_seq = ~_is_blank_series(df["junction_aa"])
    seq = df[is_seq].copy()
    seq = seq[seq["dataset"].astype(str).str.startswith("train_dataset_")].copy()
    if seq.empty:
        return {}

    seq["rank"] = _extract_rank(seq["ID"])
    seq = seq.dropna(subset=["rank"]).copy()
    seq["rank"] = seq["rank"].astype(int)
    seq = seq[seq["rank"] > 0].copy()

    out: Dict[str, Set[str]] = {}
    for ds, group in seq.groupby("dataset", observed=True):
        group = group.sort_values("rank").head(top_n)
        out[str(ds)] = set(_seq_key(group).tolist())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task-2 stability proxy via top-N Jaccard overlap.")
    p.add_argument("--submissions", nargs="+", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=10000)
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")

    sources = []
    for p in args.submissions:
        if not p.exists():
            raise FileNotFoundError(f"submission not found: {p}")
        sources.append(p)

    per_source: List[Dict[str, Set[str]]] = [load_topn_sets(p, top_n=args.top_n) for p in sources]
    train_datasets = sorted(set().union(*[set(d.keys()) for d in per_source if d]))

    rows = []
    for ds in train_datasets:
        sets = [d.get(ds, set()) for d in per_source]
        sizes = [len(s) for s in sets]

        pair_scores = []
        for (i, a), (j, b) in combinations(list(enumerate(sets)), 2):
            pair_scores.append(_jaccard(a, b))

        union_all = set().union(*sets) if sets else set()
        inter_all = set.intersection(*sets) if sets and all(sets) else set()

        rows.append(
            {
                "train_dataset": ds,
                "n_sources": len(sources),
                "top_n": args.top_n,
                "mean_set_size": float(np.mean(sizes)) if sizes else 0.0,
                "min_set_size": int(min(sizes)) if sizes else 0,
                "max_set_size": int(max(sizes)) if sizes else 0,
                "union_size": int(len(union_all)),
                "intersection_size": int(len(inter_all)),
                "mean_pairwise_jaccard": float(np.mean(pair_scores)) if pair_scores else float("nan"),
                "min_pairwise_jaccard": float(np.min(pair_scores)) if pair_scores else float("nan"),
                "max_pairwise_jaccard": float(np.max(pair_scores)) if pair_scores else float("nan"),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["mean_pairwise_jaccard", "train_dataset"], ascending=[False, True])
    pd.set_option("display.max_columns", 50)
    print(summary.to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nWrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

