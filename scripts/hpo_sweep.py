"""
Unified HPO sweep harness for tabular and deep models using Optuna.

Usage:
  python3 scripts/hpo_sweep.py --config configs/hpo_unified.json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

import numpy as np
import optuna
import pandas as pd

from airrml import config as default_config
from airrml import data
from airrml import evaluation
from airrml.utils import ensure_dir, seed_everything


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs")
        with path.open("r") as f:
            return yaml.safe_load(f)
    with path.open("r") as f:
        return json.load(f)


def suggest_params(trial: optuna.trial.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for name, spec in space.items():
        typ = spec.get("type")
        if typ == "loguniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif typ == "uniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif typ == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif typ == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search type: {typ}")
    return params


def run_study(
    model_name: str,
    model_space: Dict[str, Any],
    sequences_df,
    metadata_df,
    feature_config: Dict[str, Any],
    cv_folds: int,
    random_state: int,
    n_trials: int,
    timeout: Optional[int],
) -> optuna.study.Study:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, model_space)
        # Basic architecture constraint for transformers
        if model_name == "deep_mil":
            model_dim = params.get("model_dim")
            num_heads = params.get("num_heads")
            if model_dim is not None and num_heads is not None and int(model_dim) % int(num_heads) != 0:
                raise optuna.TrialPruned("model_dim must be divisible by num_heads")
        res = evaluation.cross_validate_model(
            model_name=model_name,
            sequences_df=sequences_df,
            label_df=metadata_df,
            feature_config=feature_config,
            cv_folds=cv_folds,
            random_state=random_state,
            model_params=params,
        )
        auc = res.get("mean_auc", float("nan"))
        return 0.0 if auc is None or np.isnan(auc) else float(auc)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True, catch=(Exception,))
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified HPO sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trial-offset", type=int, default=0, help="Offset for array jobs to split trials")
    parser.add_argument("--trial-count", type=int, default=None, help="Override number of trials for this invocation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_root = Path(cfg.get("train_root", default_config.TRAIN_ROOT))
    feature_config = cfg.get("feature_config", {"use_kmers": True, "k": 3})
    model_name = cfg.get("model_name", "kmer_logreg")
    search_space = cfg.get("search_space", {})
    cv_folds = cfg.get("cv_folds", default_config.DEFAULT_NUM_FOLDS)
    random_state = cfg.get("random_state", default_config.DEFAULT_RANDOM_SEED)
    n_trials = args.trial_count or cfg.get("n_trials", 20)
    timeout = cfg.get("timeout", None)
    timeout = int(timeout) if timeout is not None else None
    datasets_filter: List[str] = cfg.get("datasets", [])
    output_dir = Path(cfg.get("output_dir", "outputs/hpo_unified"))

    seed_everything(random_state + args.trial_offset)
    dataset_map = data.list_datasets(train_root, default_config.TEST_ROOT)
    selected = {k: v for k, v in dataset_map.items() if not datasets_filter or k in datasets_filter}

    rows = []
    for ds, info in selected.items():
        sequences_df, metadata_df = data.load_full_dataset(info["train_path"])
        study = run_study(
            model_name=model_name,
            model_space=search_space,
            sequences_df=sequences_df,
            metadata_df=metadata_df,
            feature_config=feature_config,
            cv_folds=cv_folds,
            random_state=random_state + args.trial_offset,
            n_trials=n_trials,
            timeout=timeout,
        )
        ds_out = output_dir / ds / f"trial_offset_{args.trial_offset}"
        ensure_dir(ds_out)
        study.trials_dataframe().to_csv(ds_out / "trials.csv", index=False)
        best_params_path = ds_out / "best_params.json"
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best_params = study.best_params
            best_value = study.best_value
        else:
            best_params = {}
            best_value = float("nan")
        with best_params_path.open("w") as f:
            json.dump(best_params, f, indent=2)
        rows.append({"dataset": ds, "best_auc": best_value, "best_params_path": str(best_params_path)})
        print(f"{ds}: best AUC {best_value}")

    ensure_dir(output_dir)
    pd.DataFrame(rows).to_csv(output_dir / f"run_summary_offset_{args.trial_offset}.csv", index=False)


if __name__ == "__main__":
    main()
