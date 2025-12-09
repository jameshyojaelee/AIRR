"""
L1-regularized logistic regression on k-mer repertoire features.

This model searches over C values with stratified CV, then provides both
repertoire-level probabilities and sequence-level importance scores by
projecting sequences into the trained k-mer space.
"""
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from airrml.features import apply_kmer_features
from airrml.models import register_model
from airrml.models.base import BaseRepertoireModel


@register_model("kmer_logreg")
class KmerLogReg(BaseRepertoireModel):
    """
    K-mer + L1 logistic regression repertoire classifier.
    """

    def __init__(
        self,
        c_grid: Optional[List[float]] = None,
        cv_folds: int = 0,
        opt_metric: str = "roc_auc",
        penalty: str = "l1",
        random_state: int = 123,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(c_grid=c_grid, cv_folds=cv_folds, opt_metric=opt_metric, penalty=penalty, random_state=random_state, n_jobs=n_jobs, **kwargs)
        self.c_grid = c_grid if c_grid is not None else [1.0]
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.penalty = penalty
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_: Optional[Pipeline] = None
        self.cv_results_: Optional[pd.DataFrame] = None
        self.best_C_: Optional[float] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "KmerLogReg":
        """
        Fit L1 logistic regression with C tuned via stratified CV.
        """
        self.feature_names_ = list(X_train.columns)

        solver = "liblinear" if self.penalty == "l1" else "lbfgs"
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        penalty=self.penalty,
                        solver=solver,
                        max_iter=2000,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

        if self.cv_folds and self.cv_folds > 1 and len(self.c_grid) > 1:
            param_grid = {"clf__C": self.c_grid}
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=self.opt_metric,
                n_jobs=self.n_jobs,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            self.model_ = search.best_estimator_
            self.best_C_ = search.best_params_.get("clf__C")
            self.cv_results_ = pd.DataFrame(search.cv_results_)
        else:
            # Single fit without CV to reduce runtime
            pipeline.set_params(clf__C=self.c_grid[0])
            pipeline.fit(X_train, y_train)
            self.model_ = pipeline
            self.best_C_ = self.c_grid[0]
            self.cv_results_ = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")
        if self.feature_names_ is None:
            raise RuntimeError("feature_names_ missing; call fit first.")

        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        probs = self.model_.predict_proba(X_aligned)[:, 1]
        return probs

    def get_sequence_importance(
        self,
        sequences_df: pd.DataFrame,
        sequence_col: str = "junction_aa",
    ) -> pd.DataFrame:
        """
        Project sequences into the k-mer space and compute decision scores.
        """
        if self.model_ is None or self.feature_info is None:
            raise RuntimeError("Model and feature_info must be set before scoring sequences.")
        if "kmer" not in self.feature_info:
            raise ValueError("feature_info must contain 'kmer' metadata for sequence scoring.")
        if sequences_df.empty:
            return pd.DataFrame(columns=["junction_aa", "v_call", "j_call", "importance_score"])

        seqs = sequences_df.copy().reset_index(drop=True)
        seqs["ID"] = seqs.index.astype(str)

        X_seq = apply_kmer_features(seqs, self.feature_info["kmer"], sequence_col=sequence_col)
        X_seq = X_seq.reindex(columns=self.feature_names_, fill_value=0)

        # Use pipeline decision function to stay consistent with scaling.
        scores = self.model_.decision_function(X_seq)
        seqs_out = sequences_df.copy().reset_index(drop=True)
        seqs_out["importance_score"] = scores

        # Ensure required columns exist
        for col in ("junction_aa", "v_call", "j_call"):
            if col not in seqs_out.columns:
                seqs_out[col] = None

        return seqs_out[["junction_aa", "v_call", "j_call", "importance_score"]]
