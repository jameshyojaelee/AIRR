#!/usr/bin/env python3
"""Automatically ensemble the latest successful submissions in `outputs/`.

This is a convenience wrapper around `scripts/ensemble_submissions.py`.
It finds the most recent directory matching a set of patterns that contains
`submission.csv`, then produces an ensembled submission and validates it.

Usage:
  python3 scripts/run_auto_ensemble.py
  python3 scripts/run_auto_ensemble.py --output outputs/auto_ensemble/submission.csv
"""
import argparse
import datetime as _dt
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

def find_latest_submission(output_root: Path, pattern: str) -> Optional[Path]:
    """Find the most recent directory matching the pattern that contains a submission.csv."""
    candidates = sorted(output_root.glob(pattern), key=os.path.getmtime, reverse=True)
    for c in candidates:
        sub_path = c / "submission.csv"
        if sub_path.exists():
            return sub_path
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output CSV path (default: outputs/auto_ensemble-<ts>/submission.csv)")
    parser.add_argument("--dedup-near", action="store_true", help="Collapse near-duplicate sequences for Task 2")
    parser.add_argument("--strategy", default=None, help="Optional strategy JSON passed to ensemble_submissions.py")
    parser.add_argument("--task2-method", choices=["rrf", "borda"], default="rrf")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=50000)
    args = parser.parse_args()

    repo_root = Path(os.environ.get("REPO_ROOT", "."))
    output_root = repo_root / "outputs"

    # Define the experiment types we want to merge.
    # Format: (pattern, weight_task1, weight_task2).
    # These are starting points; override by calling `scripts/ensemble_submissions.py` directly.
    experiments = [
        ("enrichment_bayes-*", 0.15, 1.0),       # New Bayesian Enrichment (Task 2)
        ("gbm_publicness-*", 1.0, 0.0),          # GBM (Task 1)
        ("deepmil_transformer-*", 0.5, 0.5),     # DeepMIL (Task 1 + Task 2)
    ]

    found_submissions: List[Tuple[Path, float, float]] = []

    print(f"Scanning {output_root} for submissions...")
    for pattern, w1, w2 in experiments:
        path = find_latest_submission(output_root, pattern)
        if path:
            print(f"  Found {pattern}: {path} (w_task1={w1}, w_task2={w2})")
            found_submissions.append((path, float(w1), float(w2)))
        else:
            print(f"  WARNING: No successful run found for {pattern}")

    if not found_submissions:
        print("No submissions found. Exiting.")
        sys.exit(1)

    paths = [str(p[0]) for p in found_submissions]
    weights_task1 = [str(p[1]) for p in found_submissions]
    weights_task2 = [str(p[2]) for p in found_submissions]

    if args.output:
        out_path = Path(args.output)
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = output_root / f"auto_ensemble-{ts}" / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python3", "scripts/ensemble_submissions.py",
        "--submissions", *paths,
        "--weights-task1", *weights_task1,
        "--weights-task2", *weights_task2,
        "--task2-method", args.task2_method,
        "--rrf-k", str(args.rrf_k),
        "--top-k", str(args.top_k),
        "--output", str(out_path),
    ]
    if args.dedup_near:
        cmd.append("--dedup-near")
    if args.strategy:
        cmd.extend(["--strategy", str(args.strategy)])
    
    print("\nRunning ensemble command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    # Validation
    print("\nValidating ensemble submission...")
    check_cmd = ["python3", "scripts/check_submission.py", "--submission", str(out_path)]
    subprocess.check_call(check_cmd)
    
    print(f"\nSUCCESS: Ensembled submission saved to {out_path}")

if __name__ == "__main__":
    main()
