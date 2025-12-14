#!/usr/bin/env python3
"""
Automatically find the latest run for each experiment type and ensemble them.

Usage:
  python3 scripts/run_auto_ensemble.py
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

def find_latest_submission(output_root: Path, pattern: str) -> Path:
    """Find the most recent directory matching the pattern that contains a submission.csv."""
    candidates = sorted(output_root.glob(pattern), key=os.path.getmtime, reverse=True)
    for c in candidates:
        sub_path = c / "submission.csv"
        if sub_path.exists():
            return sub_path
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission.csv", help="Output filename")
    args = parser.parse_args()

    repo_root = Path(os.environ.get("REPO_ROOT", "."))
    output_root = repo_root / "outputs"

    # Define the experiment types we want to merge
    # Format: (pattern, weight)
    # Weights are initial guesses; can be tuned.
    experiments = [
        ("enrichment_run1-*", 1.0),       # Strong baseline for Task 2
        ("gbm_publicness-*", 1.0),      # Strong tabular Task 1
        ("deepmil_finetune_v1-*", 1.0),    # Deep model Task 1 + Attention Sequence Ranking
    ]

    found_submissions: List[Tuple[Path, float]] = []

    print(f"Scanning {output_root} for submissions...")
    for pattern, weight in experiments:
        path = find_latest_submission(output_root, pattern)
        if path:
            print(f"  Found {pattern}: {path} (weight={weight})")
            found_submissions.append((path, weight))
        else:
            print(f"  WARNING: No successful run found for {pattern}")

    if not found_submissions:
        print("No submissions found. Exiting.")
        sys.exit(1)

    paths = [str(p[0]) for p in found_submissions]
    weights = [str(p[1]) for p in found_submissions]
    
    cmd = [
        "python3", "scripts/ensemble_submissions.py",
        "--submissions", *paths,
        "--weights", *weights,
        "--output", args.output
    ]
    
    print("\nRunning ensemble command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    # Validation
    print("\nValidating ensemble submission...")
    check_cmd = ["python3", "scripts/check_submission.py", "--submission", args.output]
    subprocess.check_call(check_cmd)
    
    print(f"\nSUCCESS: Ensembled submission saved to {args.output}")

if __name__ == "__main__":
    main()
