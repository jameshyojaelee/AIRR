"""
CLI to iterate over all training datasets and optionally produce submissions.
"""
import argparse
from pathlib import Path

from airrml import config
from airrml import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models across all AIRR-ML-25 datasets")
    parser.add_argument("--model", type=str, default="kmer_logreg", help="Registered model name")
    parser.add_argument("--train_root", type=Path, default=config.TRAIN_ROOT, help="Root directory of train_datasets")
    parser.add_argument("--test_root", type=Path, default=config.TEST_ROOT, help="Root directory of test_datasets")
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR, help="Where to store artifacts")
    parser.add_argument("--run_cross_dataset_cv", action="store_true", help="Leave-one-dataset-out evaluation")
    parser.add_argument("--cv_folds", type=int, default=5, help="Within-dataset CV folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_config = {}
    training_config = {
        "output_dir": args.output_dir,
        "cv_folds": args.cv_folds,
        "test_root": args.test_root,
    }

    if args.run_cross_dataset_cv:
        pipeline.cross_validate_across_datasets(
            model_name=args.model,
            train_root=args.train_root,
            feature_config=feature_config,
            training_config=training_config,
        )
    else:
        pipeline.generate_submission(
            model_name=args.model,
            train_root=args.train_root,
            test_root=args.test_root,
            feature_config=feature_config,
            training_config=training_config,
        )


if __name__ == "__main__":
    main()
