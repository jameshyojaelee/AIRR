"""
CLI to train and evaluate a single AIRR-ML-25 dataset.

This script wires together data loading, feature building, model training,
and optional within-dataset cross-validation.
"""
import argparse
from pathlib import Path

from airrml import config
from airrml import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on a single AIRR-ML-25 dataset")
    parser.add_argument("--model", type=str, default="kmer_logreg", help="Registered model name")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to train_dataset_X folder")
    parser.add_argument("--test_paths", type=Path, nargs="*", default=None, help="Associated test_dataset paths")
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR, help="Where to store artifacts")
    parser.add_argument("--run_cv", action="store_true", help="Whether to run within-dataset CV")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds for classical models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_config = {}
    training_config = {
        "output_dir": args.output_dir,
        "cv_folds": args.cv_folds if args.run_cv else 0,
    }

    if args.run_cv:
        pipeline.cross_validate_single_dataset(
            model_name=args.model,
            train_path=args.dataset_path,
            feature_config=feature_config,
            training_config=training_config,
        )

    if args.test_paths:
        pipeline.predict_on_test_dataset(
            model_name=args.model,
            train_dataset_path=args.dataset_path,
            test_dataset_paths=list(args.test_paths),
            feature_config=feature_config,
            training_config=training_config,
        )
    else:
        pipeline.train_model_on_dataset(
            model_name=args.model,
            train_path=args.dataset_path,
            feature_config=feature_config,
            training_config=training_config,
        )


if __name__ == "__main__":
    main()
