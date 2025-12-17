"""
Gradient boosting repertoire classifier with flexible backends.

Prefers XGBoost if available, otherwise LightGBM, and falls back to
sklearn's HistGradientBoostingClassifier. Provides approximate sequence-level
importance by projecting sequences into the k-mer feature space and combining
k-mer importances.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from airrml.features import apply_kmer_features
from airrml.models import register_model
from airrml.models.base import BaseRepertoireModel


def _detect_backend() -> Tuple[str, Optional[Any]]:
    try:
        import xgboost as xgb  # type: ignore

        return "xgb", xgb
    except ImportError:
        pass
    try:
        import lightgbm as lgb  # type: ignore

        return "lgbm", lgb
    except ImportError:
        pass
    return "sklearn", None


@register_model("gbm")
class GradientBoostingRepertoireModel(BaseRepertoireModel):
    """
    Gradient boosting repertoire model (XGBoost/LightGBM/HistGB)
    suitable for repertoire-level k-mer or other tabular features.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: Optional[int] = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds

        self.model_: Optional[Any] = None
        self.backend_: Optional[str] = None
        self.feature_names_: Optional[List[str]] = None

    def _init_model(self, X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> None:
        backend, lib = _detect_backend()
        self.backend_ = backend

        if backend == "xgb":
            xgb = lib
            self.model_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth if self.max_depth is not None else 6,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tree_method="hist",
            )
            return

        if backend == "lgbm":
            lgb = lib
            self.model_ = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=-1 if self.max_depth is None else self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="binary",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            return

        # Fallback: sklearn HistGradientBoostingClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model_ = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.n_estimators,
            random_state=self.random_state,
            early_stopping=False,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "GradientBoostingRepertoireModel":
        """
        Train the gradient boosting model; uses early stopping if the backend supports it.
        """
        self.feature_names_ = list(X_train.columns)
        self._init_model(X_val, y_val)

        if self.backend_ == "xgb":
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
                fit_kwargs["verbose"] = False
            self.model_.fit(X_train, y_train, **fit_kwargs)
        elif self.backend_ == "lgbm":
            fit_kwargs = {"eval_metric": "auc"}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
                fit_kwargs["verbose"] = False
            self.model_.fit(X_train, y_train, **fit_kwargs)
        else:
            # sklearn fallback (no explicit val set support here)
            self.model_.fit(X_train, y_train)

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
        Approximate sequence importance by weighting sequence k-mers with feature importances.
        """
        if self.model_ is None or self.feature_info is None:
            raise RuntimeError("Model and feature_info must be set before scoring sequences.")
        if "kmer" not in self.feature_info:
            raise ValueError("feature_info must contain 'kmer' metadata for sequence scoring.")
        if sequences_df.empty:
            return pd.DataFrame(columns=["junction_aa", "v_call", "j_call", "importance_score"])

        # Get feature importances aligned to training feature order
        if hasattr(self.model_, "feature_importances_"):
            importances = np.asarray(self.model_.feature_importances_, dtype=float)
        else:
            importances = np.zeros(len(self.feature_names_))
        if importances.shape[0] != len(self.feature_names_):
            importances = np.resize(importances, len(self.feature_names_))
        importance_vec = pd.Series(importances, index=self.feature_names_).fillna(0.0).to_numpy()

        seqs = sequences_df.copy().reset_index(drop=True)
        seqs["ID"] = seqs.index.astype(str)

        X_seq = apply_kmer_features(seqs, self.feature_info["kmer"], sequence_col=sequence_col)
        X_seq = X_seq.reindex(columns=self.feature_names_, fill_value=0)

        scores = X_seq.to_numpy() @ importance_vec
        seqs_out = sequences_df.copy().reset_index(drop=True)
        seqs_out["importance_score"] = scores

        for col in ("junction_aa", "v_call", "j_call"):
            if col not in seqs_out.columns:
                seqs_out[col] = None

        return seqs_out[["junction_aa", "v_call", "j_call", "importance_score"]]
