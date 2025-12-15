#!/usr/bin/env python3
"""Validate AIRR-ML submission.csv against must-pass criteria.

Checks:
- Required columns present (exact match).
- No NaNs in required fields.
- Repertoire rows: probability in [0,1], sequence fields blank.
- Sequence rows: probability placeholder (-999), junction_aa present.
- No duplicate repertoire IDs per dataset.
- No duplicate sequences per training dataset (junction_aa, v_call, j_call).
- Optional: validate against on-disk train/test roots for completeness.

Usage:
  python3 scripts/check_submission.py --submission outputs/.../submission.csv \
      --train-root /path/to/train_datasets --test-root /path/to/test_datasets/test_datasets

Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from airrml import config


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def _is_blank_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).eq("")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate AIRR-ML submission.csv")
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--test-root", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=50000, help="Max sequences per training dataset")
    p.add_argument("--allow-fewer-seqs", action="store_true", help="Allow 0 sequences for a dataset (default requires >0)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.submission.exists():
        _fail(f"submission not found: {args.submission}")
        return 1

    # Treat blank fields as empty strings, not NaN (submission schema expects blanks for sequence cols on repertoire rows).
    df = pd.read_csv(args.submission, keep_default_na=False)

    # Schema
    cols = list(df.columns)
    expected = config.SUBMISSION_COLUMNS
    if cols != expected:
        _fail(f"submission columns mismatch. Got {cols}, expected {expected}")
        return 1
    _ok("columns match submission schema")

    if df.empty:
        _fail("submission is empty")
        return 1

    # NaN checks
    if df[expected].isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        _fail(f"submission contains NaNs in columns: {nan_cols}")
        return 1
    _ok("no NaNs")

    # Row types
    is_seq_row = ~_is_blank_series(df["junction_aa"])
    is_rep_row = ~is_seq_row

    n_seq = int(is_seq_row.sum())
    n_rep = int(is_rep_row.sum())
    _ok(f"rows: {len(df)} total ({n_rep} repertoire, {n_seq} sequence)")

    # Repertoire row checks
    rep = df[is_rep_row]
    if not rep.empty:
        probs = rep["label_positive_probability"].astype(float)
        bad = (~probs.between(0.0, 1.0)).sum()
        if bad:
            _fail(f"{bad} repertoire rows have probability outside [0,1]")
            return 1
        # sequence cols must be blank
        for c in ("junction_aa", "v_call", "j_call"):
            if not _is_blank_series(rep[c]).all():
                _fail(f"repertoire rows must have blank {c}")
                return 1
        # duplicates per dataset
        dup = rep.duplicated(subset=["dataset", "ID"]).sum()
        if dup:
            _fail(f"duplicate repertoire IDs per dataset: {dup}")
            return 1
        _ok("repertoire rows valid")

    # Sequence row checks
    seq = df[is_seq_row]
    if not seq.empty:
        probs = seq["label_positive_probability"].astype(float)
        placeholder = float(config.PROBABILITY_PLACEHOLDER)
        bad = (probs != placeholder).sum()
        if bad:
            _fail(f"{bad} sequence rows have label_positive_probability != {placeholder}")
            return 1
        # must have sequence fields
        if _is_blank_series(seq["junction_aa"]).any():
            _fail("sequence rows must have non-empty junction_aa")
            return 1
        # duplicates per training dataset
        dup = seq.duplicated(subset=["dataset", "junction_aa", "v_call", "j_call"]).sum()
        if dup:
            _fail(f"duplicate sequences within dataset: {dup}")
            return 1
        # cap per dataset
        counts = seq.groupby("dataset", observed=True).size()
        over = counts[counts > args.top_k]
        if not over.empty:
            _fail(f"sequence rows exceed top_k in datasets: {over.to_dict()}")
            return 1
        if not args.allow_fewer_seqs:
            zero = counts[counts <= 0]
            if not zero.empty:
                _fail(f"datasets with 0 sequences (use --allow-fewer-seqs to permit): {zero.index.tolist()}")
                return 1
        _ok("sequence rows valid")

    # Optional on-disk validation
    if args.train_root is not None and args.test_root is not None:
        from airrml import data as data_utils

        dataset_map = data_utils.list_datasets(args.train_root, args.test_root)

        # Validate repertoire predictions match test files
        rep_by_ds = rep.groupby("dataset", observed=True)["ID"].apply(set) if not rep.empty else {}
        missing = []
        extra = []
        for train_ds, info in dataset_map.items():
            for test_path in info["test_paths"]:
                test_name = Path(test_path).name
                seq_df, meta_df = data_utils.load_test_dataset(test_path)
                expected_ids = set(meta_df["ID"].astype(str))
                got_ids = set(rep_by_ds.get(test_name, set()))
                if expected_ids - got_ids:
                    missing.append((test_name, len(expected_ids - got_ids)))
                if got_ids - expected_ids:
                    extra.append((test_name, len(got_ids - expected_ids)))
        if missing:
            _fail(f"missing repertoire predictions for: {missing}")
            return 1
        if extra:
            _fail(f"extra repertoire predictions for: {extra}")
            return 1
        _ok("repertoire predictions match test dataset IDs")

        # Validate sequence rows only for train_dataset_* names
        if not seq.empty:
            train_names = set(dataset_map.keys())
            bad_ds = sorted(set(seq["dataset"]) - train_names)
            if bad_ds:
                _fail(f"sequence rows have unexpected dataset names (should be train_dataset_*): {bad_ds}")
                return 1
            _ok("sequence rows dataset names look valid")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
