#!/usr/bin/env python3
"""Optuna sweep for enrichment hyperparams using a Task-2 proxy objective.

We optimize a combined objective:
  score = stability_weight * mean_pairwise_jaccard(topN across CV folds)
        + auc_weight       * mean_auc

This targets Task 2 list *stability* (proxy for recoverability) while keeping
Task 1 AUC from collapsing.

Usage:
  python3 scripts/enrichment_task2_sweep.py --config configs/enrichment_sweep_task2.json
  # Slurm arrays:
  python3 scripts/enrichment_task2_sweep.py --config ... --trial-offset 0 --trial-count 10
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airrml import config as default_config  # noqa: E402
from airrml import data  # noqa: E402
from airrml.utils import ensure_dir, seed_everything  # noqa: E402


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
    params: Dict[str, Any] = {}
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


def _sequence_key(df: pd.DataFrame) -> pd.Series:
    return df["junction_aa"].astype(str) + "|" + df["v_call"].astype(str) + "|" + df["j_call"].astype(str)


def _mean_pairwise_jaccard(sets: Sequence[set[str]]) -> float:
    if len(sets) < 2:
        return float("nan")
    vals: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a = sets[i]
            b = sets[j]
            if not a and not b:
                vals.append(1.0)
                continue
            if not a or not b:
                vals.append(0.0)
                continue
            inter = len(a & b)
            uni = len(a | b)
            vals.append(float(inter / uni) if uni else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _log_odds_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, alpha: float) -> np.ndarray:
    return np.log(a + alpha) + np.log(d + alpha) - np.log(b + alpha) - np.log(c + alpha)


def _maybe_subsample_presence(
    presence_df: pd.DataFrame,
    max_sequences_per_rep: Optional[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_sequences_per_rep is None:
        return presence_df
    parts = []
    for rep_id, group in presence_df.groupby("ID", observed=True):
        if len(group) > max_sequences_per_rep:
            idx = rng.choice(len(group), size=max_sequences_per_rep, replace=False)
            parts.append(group.iloc[idx])
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else presence_df.iloc[:0]


def _fit_enrichment_and_score(
    presence_df: pd.DataFrame,
    labels: pd.Series,
    params: Dict[str, Any],
    cv_folds: int,
    random_state: int,
    top_n_stability: int,
) -> Tuple[float, float]:
    """Return (mean_auc, mean_pairwise_jaccard_topN) for a parameter set."""
    alpha = float(params.get("alpha", 0.5))
    min_total_presence = int(params.get("min_total_presence", 2))
    min_pos_presence = int(params.get("min_pos_presence", 0))
    positive_only = bool(params.get("positive_only", True))
    top_k_model_seqs = int(params.get("top_k_model_seqs", 20000))
    max_sequences_per_rep = params.get("max_sequences_per_rep", None)
    if max_sequences_per_rep is not None:
        max_sequences_per_rep = int(max_sequences_per_rep)

    class_weight = params.get("class_weight", "balanced")
    if class_weight in ("none", "null"):
        class_weight = None

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    aucs: List[float] = []
    fold_sets: List[set[str]] = []
    rng = np.random.default_rng(random_state)

    y = labels.astype(int)
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        train_ids = y.index[train_idx]
        val_ids = y.index[val_idx]

        train_presence = presence_df[presence_df["ID"].isin(train_ids)]
        val_presence = presence_df[presence_df["ID"].isin(val_ids)]

        # Optional per-repertoire subsampling to control runtime/variance.
        train_presence = _maybe_subsample_presence(train_presence, max_sequences_per_rep=max_sequences_per_rep, rng=rng)
        val_presence = _maybe_subsample_presence(val_presence, max_sequences_per_rep=max_sequences_per_rep, rng=rng)

        train_labels = y.loc[train_ids]
        n_pos = float((train_labels == 1).sum())
        n_neg = float((train_labels == 0).sum())
        if n_pos <= 0 or n_neg <= 0:
            return float("nan"), float("nan")

        df = train_presence.merge(train_labels.rename("label_positive"), left_on="ID", right_index=True, how="inner")
        counts = df.groupby(["junction_aa", "v_call", "j_call", "label_positive"], observed=True).size().unstack(fill_value=0)
        pos_present = counts.get(1, pd.Series(0, index=counts.index)).to_numpy(dtype=float)
        neg_present = counts.get(0, pd.Series(0, index=counts.index)).to_numpy(dtype=float)

        pos_absent = n_pos - pos_present
        neg_absent = n_neg - neg_present

        lor = _log_odds_ratio(pos_present, neg_present, pos_absent, neg_absent, alpha=alpha)
        support = np.sqrt(pos_present + neg_present)
        score = lor * support

        stats = counts.copy().rename(columns={0: "neg_present", 1: "pos_present"}).reset_index()
        stats["neg_present"] = stats.get("neg_present", 0).astype(int)
        stats["pos_present"] = stats.get("pos_present", 0).astype(int)
        stats["importance_score"] = score

        stats = stats[(stats["pos_present"] + stats["neg_present"]) >= min_total_presence]
        stats = stats[stats["pos_present"] >= min_pos_presence]
        if positive_only:
            stats = stats[stats["importance_score"] > 0]

        stats = stats.sort_values("importance_score", ascending=False).reset_index(drop=True)
        # Fold Task-2 set
        fold_top = stats[["junction_aa", "v_call", "j_call"]].drop_duplicates().head(top_n_stability)
        fold_sets.append(set(_sequence_key(fold_top).tolist()))

        selected = stats.head(top_k_model_seqs)[["junction_aa", "v_call", "j_call", "importance_score"]].copy()
        # Build simple repertoire features for calibration (sum/max/hits)
        merged = train_presence.merge(selected, on=["junction_aa", "v_call", "j_call"], how="inner")
        rep_train = (
            merged.groupby("ID", observed=True)["importance_score"]
            .agg(score_sum="sum", score_max="max", hits="count")
            .astype(float)
            .fillna(0.0)
        )
        rep_train = rep_train.reindex(train_ids, fill_value=0.0)

        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(rep_train, train_labels.loc[rep_train.index])

        merged_val = val_presence.merge(selected, on=["junction_aa", "v_call", "j_call"], how="inner")
        rep_val = (
            merged_val.groupby("ID", observed=True)["importance_score"]
            .agg(score_sum="sum", score_max="max", hits="count")
            .astype(float)
            .fillna(0.0)
        )
        rep_val = rep_val.reindex(val_ids, fill_value=0.0)

        probs = clf.predict_proba(rep_val)[:, 1]
        try:
            aucs.append(float(roc_auc_score(y.loc[val_ids].to_numpy(), probs)))
        except ValueError:
            aucs.append(float("nan"))

    mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
    mean_j = _mean_pairwise_jaccard(fold_sets)
    return mean_auc, mean_j


def run_study(
    dataset_name: str,
    train_path: Path,
    model_space: Dict[str, Any],
    cv_folds: int,
    random_state: int,
    n_trials: int,
    timeout: Optional[int],
    top_n_stability: int,
    stability_weight: float,
    auc_weight: float,
    max_repertoires: Optional[int] = None,
) -> optuna.study.Study:
    sequences_df, metadata_df = data.load_full_dataset(train_path)

    labels_df = metadata_df.set_index("ID")
    y = labels_df[default_config.LABEL_COL].astype(int)
    if max_repertoires is not None and len(y) > max_repertoires:
        # Stratified downsample repertoires for cheaper sweeps.
        pos = y[y == 1].sample(n=max_repertoires // 2, random_state=random_state, replace=False) if (y == 1).sum() else y.iloc[:0]
        neg = y[y == 0].sample(n=max_repertoires - len(pos), random_state=random_state, replace=False) if (y == 0).sum() else y.iloc[:0]
        keep_ids = set(pos.index) | set(neg.index)
        y = y.loc[list(keep_ids)]
        sequences_df = sequences_df[sequences_df["ID"].isin(keep_ids)]

    # Precompute repertoire-level presence table once per dataset (major speedup for ds7/8).
    presence_df = sequences_df[["ID", "junction_aa", "v_call", "j_call"]].dropna(subset=["ID", "junction_aa"]).copy()
    presence_df["ID"] = presence_df["ID"].astype(str)
    for c in ("junction_aa", "v_call", "j_call"):
        presence_df[c] = presence_df[c].fillna("").astype(str)
    presence_df = presence_df.drop_duplicates(subset=["ID", "junction_aa", "v_call", "j_call"])

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, model_space)
        start = time.time()
        mean_auc, mean_j = _fit_enrichment_and_score(
            presence_df=presence_df,
            labels=y,
            params=params,
            cv_folds=cv_folds,
            random_state=random_state + trial.number,
            top_n_stability=top_n_stability,
        )
        elapsed = time.time() - start
        trial.set_user_attr("mean_auc", mean_auc)
        trial.set_user_attr("mean_jaccard", mean_j)
        trial.set_user_attr("elapsed_sec", elapsed)
        if math.isnan(mean_j):
            raise optuna.TrialPruned("stability undefined")
        score = float(stability_weight) * float(mean_j) + float(auc_weight) * float(0.0 if math.isnan(mean_auc) else mean_auc)
        return score

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True, catch=(Exception,))
    return study


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrichment Task-2 hyperparameter sweep (Optuna)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--trial-count", type=int, default=None, help="Override number of trials for this invocation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_root = Path(cfg.get("train_root", default_config.TRAIN_ROOT))
    test_root = Path(cfg.get("test_root", default_config.TEST_ROOT))
    output_dir = Path(cfg.get("output_dir", "outputs/enrichment_sweep_task2"))
    datasets_filter: List[str] = cfg.get("datasets", []) or []
    search_space = cfg.get("search_space", {}) or {}

    cv_folds = int(cfg.get("cv_folds", default_config.DEFAULT_NUM_FOLDS))
    random_state = int(cfg.get("random_state", default_config.DEFAULT_RANDOM_SEED))
    n_trials = int(args.trial_count or cfg.get("n_trials", 20))
    timeout = cfg.get("timeout", None)
    timeout = int(timeout) if timeout is not None else None

    top_n_stability = int(cfg.get("top_n_stability", 10000))
    stability_weight = float(cfg.get("stability_weight", 1.0))
    auc_weight = float(cfg.get("auc_weight", 0.25))
    max_repertoires = cfg.get("max_repertoires", None)
    max_repertoires = int(max_repertoires) if max_repertoires is not None else None

    seed_everything(random_state + args.trial_offset)
    dataset_map = data.list_datasets(train_root, test_root)
    selected = {k: v for k, v in dataset_map.items() if not datasets_filter or k in datasets_filter}

    rows = []
    for ds, info in selected.items():
        print(f"\n=== {ds} ===")
        study = run_study(
            dataset_name=ds,
            train_path=Path(info["train_path"]),
            model_space=search_space,
            cv_folds=cv_folds,
            random_state=random_state + args.trial_offset,
            n_trials=n_trials,
            timeout=timeout,
            top_n_stability=top_n_stability,
            stability_weight=stability_weight,
            auc_weight=auc_weight,
            max_repertoires=max_repertoires,
        )

        ds_out = output_dir / ds / f"trial_offset_{args.trial_offset}"
        ensure_dir(ds_out)
        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        df.to_csv(ds_out / "trials.csv", index=False)

        best_params_path = ds_out / "best_params.json"
        best_summary_path = ds_out / "best_summary.json"

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = float(best_trial.value)
            best_auc = best_trial.user_attrs.get("mean_auc")
            best_j = best_trial.user_attrs.get("mean_jaccard")
        else:
            best_params = {}
            best_score = float("nan")
            best_auc = float("nan")
            best_j = float("nan")

        with best_params_path.open("w") as f:
            json.dump(best_params, f, indent=2)
        with best_summary_path.open("w") as f:
            json.dump({"best_score": best_score, "best_auc": best_auc, "best_jaccard": best_j}, f, indent=2)

        rows.append(
            {
                "dataset": ds,
                "best_score": best_score,
                "best_auc": best_auc,
                "best_jaccard": best_j,
                "best_params_path": str(best_params_path),
            }
        )
        print(f"{ds}: best score={best_score} auc={best_auc} jaccard={best_j}")

    ensure_dir(output_dir)
    pd.DataFrame(rows).to_csv(output_dir / f"run_summary_offset_{args.trial_offset}.csv", index=False)


if __name__ == "__main__":
    main()

