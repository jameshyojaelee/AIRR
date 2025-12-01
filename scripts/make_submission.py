"""
CLI to build submission.csv using trained models and saved artifacts.
"""
import argparse
from pathlib import Path

from airrml import config
from airrml.submission import assemble_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AIRR-ML-25 submission.csv")
    parser.add_argument("--train-root", type=Path, default=config.TRAIN_ROOT, help="Root of train_datasets")
    parser.add_argument("--test-root", type=Path, default=config.TEST_ROOT, help="Root of test_datasets")
    parser.add_argument("--model-name", type=str, default="kmer_logreg", help="Registered model name")
    parser.add_argument("--model-output-root", type=Path, default=config.OUTPUT_DIR, help="Directory with trained models per dataset")
    parser.add_argument("--top-k-sequences", type=int, default=50000, help="How many sequences to keep for Task 2")
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=config.SUBMISSION_DIR / "submission.csv",
        help="Where to write submission.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assemble_submission(
        train_root=str(args.train_root),
        test_root=str(args.test_root),
        model_name=args.model_name,
        model_output_root=str(args.model_output_root),
        top_k_sequences=args.top_k_sequences,
        submission_path=str(args.submission_path),
    )


if __name__ == "__main__":
    main()
