"""
Bayesian Sequence Enrichment Model for AIRR-ML-25.

This model robustly estimates label association statistics using Beta-Binomial shrinkage.
Instead of relying on hard thresholds or raw log-odds, it shrinks prevalence estimates
towards a global background prior, significantly reducing noise from rare sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from airrml import config
from airrml.models import register_model
from airrml.models.base import BaseRepertoireModel


@dataclass
class EnrichmentBayesConfig:
    # Beta prior strength: higher = more shrinkage towards background
    prior_strength_pos: float = 10.0 
    prior_strength_neg: float = 20.0 
    min_total_presence: int = 1
    top_k_model_seqs: int = 25000
    max_sequences_per_rep: Optional[int] = None
    class_weight: Optional[str] = "balanced"
    n_jobs: int = -1


def _validate_labels(y: pd.Series) -> pd.Series:
    y = y.copy()
    y = y.dropna()
    y = y.astype(int)
    return y


def _maybe_subsample(sequences_df: pd.DataFrame, max_sequences_per_rep: Optional[int], rng: np.random.Generator) -> pd.DataFrame:
    if max_sequences_per_rep is None:
        return sequences_df
    parts = []
    for rep_id, group in sequences_df.groupby("ID"):
        if len(group) > max_sequences_per_rep:
            idx = rng.choice(len(group), size=max_sequences_per_rep, replace=False)
            parts.append(group.iloc[idx])
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else sequences_df.iloc[:0]


def _sequence_columns(sequence_col: str = "junction_aa") -> Tuple[str, str, str]:
    return sequence_col, "v_call", "j_call"


def _compute_presence_table(sequences_df: pd.DataFrame, sequence_col: str) -> pd.DataFrame:
    seq_col, v_col, j_col = _sequence_columns(sequence_col)
    required = {"ID", seq_col, v_col, j_col}
    missing = required - set(sequences_df.columns)
    if missing:
        raise ValueError(f"sequences_df missing required columns: {missing}")

    df = sequences_df[["ID", seq_col, v_col, j_col]].dropna(subset=["ID", seq_col]).copy()
    df["ID"] = df["ID"].astype(str)
    df[seq_col] = df[seq_col].astype(str)
    df[v_col] = df[v_col].fillna("").astype(str)
    df[j_col] = df[j_col].fillna("").astype(str)
    # Presence/absence at repertoire level
    df = df.drop_duplicates(subset=["ID", seq_col, v_col, j_col])
    return df


@register_model("enrichment_bayes")
class EnrichmentBayesModel(BaseRepertoireModel):
    """
    Beta-Binomial sequence enrichment model.
    """

    consumes_sequences: bool = True

    def __init__(
        self,
        prior_strength_pos: float = 10.0,
        prior_strength_neg: float = 20.0,
        min_total_presence: int = 1,
        top_k_model_seqs: int = 25000,
        max_sequences_per_rep: Optional[int] = None,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            prior_strength_pos=prior_strength_pos,
            prior_strength_neg=prior_strength_neg,
            min_total_presence=min_total_presence,
            top_k_model_seqs=top_k_model_seqs,
            max_sequences_per_rep=max_sequences_per_rep,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.cfg = EnrichmentBayesConfig(
            prior_strength_pos=float(prior_strength_pos),
            prior_strength_neg=float(prior_strength_neg),
            min_total_presence=int(min_total_presence),
            top_k_model_seqs=int(top_k_model_seqs),
            max_sequences_per_rep=max_sequences_per_rep,
            class_weight=class_weight,
            n_jobs=int(n_jobs),
        )
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

        self.sequence_stats_: Optional[pd.DataFrame] = None
        self.selected_stats_: Optional[pd.DataFrame] = None
        self.calibrator_: Optional[LogisticRegression] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "EnrichmentBayesModel":
        labels = _validate_labels(y_train)
        presence_df = _compute_presence_table(X_train, sequence_col=config.SEQUENCE_COLS[0])
        presence_df = presence_df[presence_df["ID"].isin(labels.index)]
        presence_df = _maybe_subsample(presence_df, self.cfg.max_sequences_per_rep, self.rng)

        df = presence_df.merge(labels.rename(config.LABEL_COL), left_on="ID", right_index=True, how="inner")
        seq_col, v_col, j_col = _sequence_columns(config.SEQUENCE_COLS[0])

        counts = df.groupby([seq_col, v_col, j_col, config.LABEL_COL], observed=True).size().unstack(fill_value=0)
        pos_counts = counts.get(1, pd.Series(0, index=counts.index)).to_numpy(dtype=float)
        neg_counts = counts.get(0, pd.Series(0, index=counts.index)).to_numpy(dtype=float)
        
        n_pos = float((labels == 1).sum())
        n_neg = float((labels == 0).sum())

        # --- Bayesian Shrinkage Core ---
        # 1. Estimate global prevalence prior (mean rate across ALL sequences)
        # Often very low, like 1e-4. 
        # But we want a "background" rate.
        # Let's simply center the Beta distribution at the observed global mean rate.
        
        # Mean prevalence in any repertoire
        total_occs = pos_counts.sum() + neg_counts.sum()
        total_reps = n_pos + n_neg
        total_possible = total_reps * len(pos_counts)
        global_mean_rate = total_occs / total_possible if total_possible > 0 else 1e-5

        # Beta(alpha, beta) => mean = alpha / (alpha + beta)
        # alpha_prior = global_mean_rate * strength
        # beta_prior = (1 - global_mean_rate) * strength
        
        # We can perform independent shrinkage for Pos and Neg rates.
        
        def shrink_rate(obs_k, total_n, strength):
            mean_rate = obs_k.sum() / (total_n * len(obs_k)) + 1e-9
            alpha_0 = mean_rate * strength
            beta_0 = (1.0 - mean_rate) * strength
            # Posterior mean: (k + alpha_0) / (n + alpha_0 + beta_0)
            return (obs_k + alpha_0) / (total_n + alpha_0 + beta_0)

        # Shrink negative rate (Background) more aggressively (stronger prior)
        p_neg = shrink_rate(neg_counts, n_neg, self.cfg.prior_strength_neg)
        
        # Shrink positive rate less aggressively (allow signal to emerge)
        p_pos = shrink_rate(pos_counts, n_pos, self.cfg.prior_strength_pos)
        
        # Log Odds Ratio of shrunk probabilities
        # log ( p_pos / (1-p_pos) ) - log ( p_neg / (1-p_neg) )
        logit_pos = np.log(p_pos) - np.log(1.0 - p_pos)
        logit_neg = np.log(p_neg) - np.log(1.0 - p_neg)
        
        log_odds = logit_pos - logit_neg
        
        # "Signed Information" score: Log-Odds * Uncertainty-Aware-Weight?
        # Standard Log-Odds is already good if p_pos/p_neg are stable.
        # But let's scale by prevalence to favor widely-present motifs over singleton spikes.
        # A simple robust score:
        # score = log_odds * log10(pos_counts + neg_counts + 1)
        score = log_odds 

        # -----------------------------

        stats = counts.copy()
        stats = stats.rename(columns={0: "neg_present", 1: "pos_present"}).reset_index()
        stats["neg_present"] = stats["neg_present"].astype(int)
        stats["pos_present"] = stats["pos_present"].astype(int)
        stats["importance_score"] = score
        
        stats = stats[(stats["pos_present"] + stats["neg_present"]) >= self.cfg.min_total_presence]
        # Only keep positive associations
        stats = stats[stats["importance_score"] > 0]
        
        stats = stats.sort_values("importance_score", ascending=False).reset_index(drop=True)
        self.sequence_stats_ = stats

        selected = stats.head(self.cfg.top_k_model_seqs).copy()
        self.selected_stats_ = selected

        # Build repertoire-level features for calibration
        rep_features = self._build_repertoire_features(presence_df, selected)
        rep_features = rep_features.reindex(labels.index, fill_value=0.0)

        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight=self.cfg.class_weight,
            n_jobs=self.cfg.n_jobs,
            random_state=self.random_state,
        )
        clf.fit(rep_features, labels.loc[rep_features.index])
        self.calibrator_ = clf
        return self

    def _build_repertoire_features(self, presence_df: pd.DataFrame, selected_stats: pd.DataFrame) -> pd.DataFrame:
        seq_col, v_col, j_col = _sequence_columns(config.SEQUENCE_COLS[0])
        score_df = selected_stats[[seq_col, v_col, j_col, "importance_score"]].copy()
        merged = presence_df.merge(score_df, on=[seq_col, v_col, j_col], how="inner")

        agg = (
            merged.groupby("ID", observed=True)["importance_score"]
            .agg(score_sum="sum", score_max="max", hits="count")
            .astype(float)
            .fillna(0.0)
        )
        agg["hits"] = agg["hits"].astype(float)
        agg.index = agg.index.astype(str)
        return agg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator_ is None or self.selected_stats_ is None:
            raise RuntimeError("Model has not been fitted.")

        presence_df = _compute_presence_table(X, sequence_col=config.SEQUENCE_COLS[0])
        presence_df = _maybe_subsample(presence_df, self.cfg.max_sequences_per_rep, self.rng)
        rep_features = self._build_repertoire_features(presence_df, self.selected_stats_)

        # Align to repertoire order in X
        rep_ids = X["ID"].drop_duplicates().astype(str).tolist()
        rep_features = rep_features.reindex(rep_ids, fill_value=0.0)
        probs = self.calibrator_.predict_proba(rep_features)[:, 1]
        return probs

    def get_sequence_importance(self, sequences_df: pd.DataFrame, sequence_col: str = "junction_aa") -> pd.DataFrame:
        if self.sequence_stats_ is None:
            raise RuntimeError("Model has not been fitted.")
        seq_col, v_col, j_col = _sequence_columns(sequence_col)
        df = sequences_df.copy()
        for col in (seq_col, v_col, j_col):
            if col not in df.columns:
                df[col] = ""
        df = df[[seq_col, v_col, j_col]].copy()
        df[seq_col] = df[seq_col].fillna("").astype(str)
        df[v_col] = df[v_col].fillna("").astype(str)
        df[j_col] = df[j_col].fillna("").astype(str)

        stats = self.sequence_stats_[[seq_col, v_col, j_col, "importance_score"]].copy()
        out = df.merge(stats, on=[seq_col, v_col, j_col], how="left")
        out["importance_score"] = out["importance_score"].fillna(0.0)
        return out[[seq_col, v_col, j_col, "importance_score"]]
