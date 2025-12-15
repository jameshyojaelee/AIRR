#!/usr/bin/env python3
"""Build a hybrid AIRR-ML submission from two existing submissions.

Hybrid rule:
  - Task 1 (repertoire probabilities): taken from --task1-submission
  - Task 2 (sequence rows): taken from --task2-submission

This is useful when you trust one model for classification but another model
for sequence ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from airrml import config


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _is_blank_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).eq("")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a hybrid AIRR-ML submission.csv")
    p.add_argument("--task1-submission", type=Path, required=True, help="Submission CSV to provide repertoire probabilities")
    p.add_argument("--task2-submission", type=Path, required=True, help="Submission CSV to provide sequence rows")
    p.add_argument("--out", type=Path, required=True, help="Output submission CSV path")
    return p.parse_args()


def _read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if list(df.columns) != config.SUBMISSION_COLUMNS:
        raise ValueError(f"columns mismatch in {path}. Got {list(df.columns)} expected {config.SUBMISSION_COLUMNS}")
    return df


def main() -> int:
    args = parse_args()
    if not args.task1_submission.exists():
        return _fail(f"task1 submission not found: {args.task1_submission}")
    if not args.task2_submission.exists():
        return _fail(f"task2 submission not found: {args.task2_submission}")

    task1 = _read_submission(args.task1_submission)
    task2 = _read_submission(args.task2_submission)

    t1_seq = ~_is_blank_series(task1["junction_aa"])
    t2_seq = ~_is_blank_series(task2["junction_aa"])

    rep_rows = task1[~t1_seq].copy()
    seq_rows = task2[t2_seq].copy()

    if rep_rows.empty:
        return _fail("task1 submission has 0 repertoire rows")
    if seq_rows.empty:
        return _fail("task2 submission has 0 sequence rows")

    out = pd.concat([seq_rows, rep_rows], ignore_index=True)[config.SUBMISSION_COLUMNS]

    # Basic de-dupe safety
    rep_dup = out[_is_blank_series(out["junction_aa"])].duplicated(subset=["dataset", "ID"]).sum()
    if rep_dup:
        return _fail(f"hybrid contains duplicate repertoire IDs per dataset: {rep_dup}")
    seq_dup = out[~_is_blank_series(out["junction_aa"])].duplicated(subset=["dataset", "junction_aa", "v_call", "j_call"]).sum()
    if seq_dup:
        return _fail(f"hybrid contains duplicate sequences within dataset: {seq_dup}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote hybrid submission: {args.out}")
    print(f"  repertoire rows: {len(rep_rows)} (from {args.task1_submission})")
    print(f"  sequence rows:   {len(seq_rows)} (from {args.task2_submission})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
