"""
High-level orchestration for training, evaluation, and inference.

This module ties together data loading, feature building, model training,
and submission construction without committing to any specific model type.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from airrml import config
from airrml import data as data_utils
from airrml import evaluation
from airrml import features
from airrml import submission
from airrml import training
from airrml.models import get_model


def train_model_on_dataset(
    model_name: str,
    train_path: Path,
    feature_config: Dict,
    training_config: Dict,
) -> Tuple[object, Dict]:
    """
    Load one dataset, build features, train a model, and return artifacts.
    """
    dataset_name = Path(train_path).name
    cv_folds = training_config.get("cv_folds", config.DEFAULT_NUM_FOLDS)
    random_state = training_config.get("random_state", config.DEFAULT_RANDOM_SEED)
    output_dir = training_config.get("output_dir", config.OUTPUT_DIR)
    model_params = training_config.get("model_params", {})

    metrics = training.train_on_dataset(
        dataset_name=dataset_name,
        train_path=str(train_path),
        model_name=model_name,
        feature_config=feature_config,
        output_dir=str(output_dir),
        cv_folds=cv_folds,
        random_state=random_state,
        model_params=model_params,
    )
    return metrics, {"dataset": dataset_name, "output_dir": output_dir}


def cross_validate_single_dataset(
    model_name: str,
    train_path: Path,
    feature_config: Dict,
    training_config: Dict,
) -> Dict:
    """
    Run within-dataset cross-validation to estimate performance.
    """
    sequences_df, metadata_df = data_utils.load_full_dataset(train_path)
    cv_folds = training_config.get("cv_folds", config.DEFAULT_NUM_FOLDS)
    random_state = training_config.get("random_state", config.DEFAULT_RANDOM_SEED)
    model_params = training_config.get("model_params", {})
    return evaluation.cross_validate_model(
        model_name=model_name,
        sequences_df=sequences_df,
        label_df=metadata_df,
        feature_config=feature_config,
        cv_folds=cv_folds,
        random_state=random_state,
        model_params=model_params,
    )


def cross_validate_across_datasets(
    model_name: str,
    train_root: Path = config.TRAIN_ROOT,
    feature_config: Dict = None,
    training_config: Dict = None,
) -> Dict:
    """
    Run leave-one-dataset-out evaluation across all available datasets.
    """
    feature_config = feature_config or {"use_kmers": True, "k": 3}
    training_config = training_config or {}
    random_state = training_config.get("random_state", config.DEFAULT_RANDOM_SEED)
    model_params = training_config.get("model_params", {})
    test_root = training_config.get("test_root", config.TEST_ROOT)

    dataset_map = data_utils.list_datasets(train_root, test_root)
    results: Dict[str, Dict] = {}

    # Leave-one-dataset-out: train on all other datasets, evaluate on hold-out
    for holdout_name, holdout_info in dataset_map.items():
        train_sequences = []
        train_labels = []
        for other_name, other_info in dataset_map.items():
            if other_name == holdout_name:
                continue
            seq_df, meta_df = data_utils.load_full_dataset(other_info["train_path"])
            seq_df = seq_df.copy()
            meta_df = meta_df.copy()
            seq_df["ID"] = seq_df["ID"].apply(lambda x, n=other_name: f"{n}__{x}")
            meta_df["ID"] = meta_df["ID"].apply(lambda x, n=other_name: f"{n}__{x}")
            train_sequences.append(seq_df)
            train_labels.append(meta_df)

        if not train_sequences:
            continue

        train_seq_df = pd.concat(train_sequences, ignore_index=True)
        train_label_df = pd.concat(train_labels, ignore_index=True)

        holdout_seq_df, holdout_meta_df = data_utils.load_full_dataset(holdout_info["train_path"])
        holdout_seq_df = holdout_seq_df.copy()
        holdout_meta_df = holdout_meta_df.copy()
        holdout_seq_df["ID"] = holdout_seq_df["ID"].apply(lambda x, n=holdout_name: f"{n}__{x}")
        holdout_meta_df["ID"] = holdout_meta_df["ID"].apply(lambda x, n=holdout_name: f"{n}__{x}")

        cv_result: Dict[str, Any] = {}

        # Fit on training portion and evaluate on hold-out once
        model = get_model(model_name, random_state=random_state, **model_params)
        if model_name == "deep_mil":
            labels = train_label_df.set_index("ID")[config.LABEL_COL]
            model.fit(train_seq_df, labels)
            holdout_probs = model.predict_proba(holdout_seq_df)
            labels_holdout = holdout_meta_df.set_index("ID")[config.LABEL_COL]
            prob_series = pd.Series(holdout_probs, index=holdout_seq_df["ID"].drop_duplicates())
            holdout_probs = prob_series.reindex(labels_holdout.index).to_numpy()
        else:
            X_train, y_train, feature_info = features.build_combined_feature_matrix(train_seq_df, train_label_df, feature_config)
            model.set_feature_info(feature_info)
            model.fit(X_train, y_train)
            X_val, y_val, _ = features.build_combined_feature_matrix(holdout_seq_df, holdout_meta_df, feature_config, feature_info=feature_info)
            holdout_probs = model.predict_proba(X_val)
            labels_holdout = y_val

        auc = evaluation.compute_auc(labels_holdout.to_numpy(), holdout_probs)
        cv_result["holdout_auc"] = auc
        results[holdout_name] = cv_result

    return results


def predict_on_test_dataset(
    model_name: str,
    train_dataset_path: Path,
    test_dataset_paths: List[Path],
    feature_config: Dict,
    training_config: Dict,
) -> pd.DataFrame:
    """
    Train on full training data and produce repertoire-level predictions for test sets.
    """
    sequences_df, metadata_df = data_utils.load_full_dataset(train_dataset_path)
    random_state = training_config.get("random_state", config.DEFAULT_RANDOM_SEED)
    model_params = training_config.get("model_params", {})

    model = get_model(model_name, random_state=random_state, **model_params)
    feature_info: Optional[Dict] = None

    if model_name == "deep_mil":
        labels = metadata_df.set_index("ID")[config.LABEL_COL]
        model.fit(sequences_df, labels)
        feature_info = getattr(model, "feature_info", None)
    else:
        X_full, y_full, feature_info = features.build_combined_feature_matrix(sequences_df, metadata_df, feature_config)
        model.set_feature_info(feature_info)
        model.fit(X_full, y_full)

    all_preds = []
    for test_path in test_dataset_paths:
        test_seq_df, test_meta_df = data_utils.load_test_dataset(test_path)
        preds = submission.build_repertoire_predictions(model, feature_info or {}, test_seq_df, test_meta_df, dataset_name=test_path.name)
        all_preds.append(preds)

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(columns=config.SUBMISSION_COLUMNS)


def generate_submission(
    model_name: str,
    train_root: Path = config.TRAIN_ROOT,
    test_root: Path = config.TEST_ROOT,
    feature_config: Dict = None,
    training_config: Dict = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Train/evaluate across datasets and write a submission.csv file.
    """
    feature_config = feature_config or {"use_kmers": True, "k": 3}
    training_config = training_config or {}
    output_root = training_config.get("output_dir", config.OUTPUT_DIR)
    cv_folds = training_config.get("cv_folds", config.DEFAULT_NUM_FOLDS)
    random_state = training_config.get("random_state", config.DEFAULT_RANDOM_SEED)
    model_params = training_config.get("model_params", {})

    training.train_all_datasets(
        train_root=str(train_root),
        model_name=model_name,
        feature_config=feature_config,
        output_root=str(output_root),
        cv_folds=cv_folds,
        random_state=random_state,
        test_root=str(test_root),
        model_params=model_params,
    )

    submission_path = output_path or (config.SUBMISSION_DIR / "submission.csv")
    submission_path = Path(submission_path)
    submission.assemble_submission(
        train_root=str(train_root),
        test_root=str(test_root),
        model_name=model_name,
        model_output_root=str(output_root),
        top_k_sequences=training_config.get("top_k_sequences", 50000),
        submission_path=str(submission_path),
    )

    return submission_path
