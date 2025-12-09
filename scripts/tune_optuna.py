"""
Optuna hyperparameter tuning for AIRR-ML models using within-dataset CV.

Supports tuning kmer_logreg and gbm with multi-k TF-IDF features. Results are
saved per-dataset with a run summary CSV.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is optional
    yaml = None

import optuna
import pandas as pd

from airrml import config as default_config
from airrml import data
from airrml import evaluation
from airrml.utils import ensure_dir, seed_everything


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs")
        with path.open("r") as f:
            return yaml.safe_load(f)
    with path.open("r") as f:
        return json.load(f)


def _suggest_from_space(trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]) -> Any:
    p_type = spec.get("type")
    if p_type == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if p_type == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if p_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if p_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported search space type for {name}: {p_type}")


def _sample_params(model_name: str, trial: optuna.trial.Trial, search_space: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    space = search_space or {}
    params: Dict[str, Any] = {}

    if model_name == "kmer_logreg":
        c_spec = space.get("C", {"type": "loguniform", "low": 1e-3, "high": 10.0})
        pen_spec = space.get("penalty", {"type": "categorical", "choices": ["l1", "l2"]})
        c_val = _suggest_from_space(trial, "C", c_spec)
        penalty = _suggest_from_space(trial, "penalty", pen_spec)
        params = {"c_grid": [float(c_val)], "penalty": str(penalty), "cv_folds": 0}
    elif model_name == "gbm":
        lr_spec = space.get("learning_rate", {"type": "loguniform", "low": 0.01, "high": 0.3})
        depth_spec = space.get("max_depth", {"type": "int", "low": 3, "high": 8})
        subsample_spec = space.get("subsample", {"type": "uniform", "low": 0.6, "high": 1.0})
        colsample_spec = space.get("colsample_bytree", {"type": "uniform", "low": 0.6, "high": 1.0})
        n_estimators_spec = space.get("n_estimators", {"type": "int", "low": 200, "high": 600})

        params = {
            "learning_rate": float(_suggest_from_space(trial, "learning_rate", lr_spec)),
            "max_depth": int(_suggest_from_space(trial, "max_depth", depth_spec)),
            "subsample": float(_suggest_from_space(trial, "subsample", subsample_spec)),
            "colsample_bytree": float(_suggest_from_space(trial, "colsample_bytree", colsample_spec)),
            "n_estimators": int(_suggest_from_space(trial, "n_estimators", n_estimators_spec)),
            "n_jobs": -1,
        }
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    return params


def _tune_model_for_dataset(
    model_name: str,
    sequences_df,
    metadata_df,
    feature_config: Dict[str, Any],
    cv_folds: int,
    random_state: int,
    n_trials: int,
    timeout: Optional[int],
    search_space: Optional[Dict[str, Any]],
) -> optuna.study.Study:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        model_params = _sample_params(model_name, trial, search_space)
        results = evaluation.cross_validate_model(
            model_name=model_name,
            sequences_df=sequences_df,
            label_df=metadata_df,
            feature_config=feature_config,
            cv_folds=cv_folds,
            random_state=random_state,
            model_params=model_params,
        )
        auc = results.get("mean_auc", float("nan"))
        if auc is None or pd.isna(auc):
            return 0.0
        return float(auc)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for AIRR-ML models")
    parser.add_argument("--config", type=Path, default=Path("configs/hpo_multik_tfidf.json"), help="Path to HPO config (json/yaml)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_root = Path(cfg.get("train_root", default_config.TRAIN_ROOT))
    test_root = Path(cfg.get("test_root", default_config.TEST_ROOT))
    feature_config = cfg.get("feature_config", {"use_kmers": True, "k_list": [3, 4, 5, 6], "tfidf": True})
    hpo_cfg = cfg.get("hpo", {})
    models_cfg = cfg.get("models", [{"name": cfg.get("model_name", "kmer_logreg")}])

    cv_folds = hpo_cfg.get("cv_folds", default_config.DEFAULT_NUM_FOLDS)
    random_state = hpo_cfg.get("random_state", default_config.DEFAULT_RANDOM_SEED)
    n_trials_default = hpo_cfg.get("n_trials", 20)
    timeout = hpo_cfg.get("timeout", None)
    timeout = int(timeout) if timeout is not None else None
    datasets_filter: List[str] = hpo_cfg.get("datasets", [])
    output_root = Path(hpo_cfg.get("output_dir", "outputs/hpo"))

    seed_everything(random_state)

    dataset_map = data.list_datasets(train_root, test_root)
    selected = {k: v for k, v in dataset_map.items() if not datasets_filter or k in datasets_filter}
    if not selected:
        raise ValueError("No datasets selected for tuning (check hpo.datasets filter).")

    summary_rows: List[Dict[str, Any]] = []

    for dataset_name, info in selected.items():
        print(f"Tuning on {dataset_name}...")
        sequences_df, metadata_df = data.load_full_dataset(info["train_path"])

        for model_entry in models_cfg:
            model_name = model_entry.get("name")
            if not model_name:
                continue
            search_space = model_entry.get("search_space", {})
            n_trials = int(model_entry.get("n_trials", n_trials_default))

            study = _tune_model_for_dataset(
                model_name=model_name,
                sequences_df=sequences_df,
                metadata_df=metadata_df,
                feature_config=feature_config,
                cv_folds=cv_folds,
                random_state=random_state,
                n_trials=n_trials,
                timeout=timeout,
                search_space=search_space,
            )

            ds_out = output_root / dataset_name / model_name
            ensure_dir(ds_out)
            trials_df = study.trials_dataframe()
            trials_path = ds_out / "trials.csv"
            trials_df.to_csv(trials_path, index=False)

            best_params_path = ds_out / "best_params.json"
            with best_params_path.open("w") as f:
                json.dump(study.best_params, f, indent=2)

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "best_value_auc": study.best_value,
                    "best_params_path": str(best_params_path),
                    "trials_path": str(trials_path),
                    "n_trials": n_trials,
                    "feature_config": feature_config,
                }
            )
            print(f"  {model_name} best AUC: {study.best_value}, params: {study.best_params}")

    ensure_dir(output_root)
    summary_path = output_root / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Run summary written to: {summary_path}")


if __name__ == "__main__":
    main()
