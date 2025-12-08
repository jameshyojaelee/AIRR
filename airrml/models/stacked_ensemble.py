"""
Simple stacking ensemble that blends multiple base tabular models using a logistic regression meta-model.

Uses out-of-fold predictions from base models to train the blender, then
re-trains each base model on full data for inference.
"""
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from airrml.models import get_model, register_model
from airrml.models.base import BaseRepertoireModel


@register_model("stacked_ensemble")
class StackedEnsemble(BaseRepertoireModel):
    """
    Stack multiple base models (e.g., kmer_logreg + gbm) with a logistic regression blender.
    """

    def __init__(
        self,
        base_models: Optional[List[str]] = None,
        meta_folds: int = 5,
        random_state: int = 123,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_models=base_models, meta_folds=meta_folds, random_state=random_state, n_jobs=n_jobs, **kwargs)
        self.base_model_names = base_models or ["kmer_logreg", "gbm"]
        self.meta_folds = meta_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.base_models_: Optional[List[BaseRepertoireModel]] = None
        self.meta_model_: Optional[LogisticRegression] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "StackedEnsemble":
        self.feature_names_ = list(X_train.columns)
        y_arr = y_train.to_numpy()
        n_samples = len(y_arr)
        n_base = len(self.base_model_names)
        oof_preds = np.zeros((n_samples, n_base), dtype=float)

        if self.meta_folds and self.meta_folds > 1:
            skf = StratifiedKFold(n_splits=self.meta_folds, shuffle=True, random_state=self.random_state)
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y_arr)):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val_fold = X_train.iloc[val_idx]
                for m_idx, name in enumerate(self.base_model_names):
                    model = get_model(name, random_state=self.random_state)
                    if self.feature_info is not None and hasattr(model, "set_feature_info"):
                        model.set_feature_info(self.feature_info)
                    model.fit(X_tr, y_tr)
                    oof_preds[val_idx, m_idx] = model.predict_proba(X_val_fold)
        else:
            # Fallback: single train split (no OOF)
            for m_idx, name in enumerate(self.base_model_names):
                model = get_model(name, random_state=self.random_state)
                if self.feature_info is not None and hasattr(model, "set_feature_info"):
                    model.set_feature_info(self.feature_info)
                model.fit(X_train, y_train)
                oof_preds[:, m_idx] = model.predict_proba(X_train)

        meta_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        meta_model.fit(oof_preds, y_arr)

        # Train base models on full data for inference
        fitted_base: List[BaseRepertoireModel] = []
        for name in self.base_model_names:
            model = get_model(name, random_state=self.random_state)
            if self.feature_info is not None and hasattr(model, "set_feature_info"):
                model.set_feature_info(self.feature_info)
            model.fit(X_train, y_train)
            fitted_base.append(model)

        self.base_models_ = fitted_base
        self.meta_model_ = meta_model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.base_models_ is None or self.meta_model_ is None:
            raise RuntimeError("StackedEnsemble has not been fitted.")
        if self.feature_names_ is None:
            raise RuntimeError("feature_names_ missing; call fit first.")

        base_pred_blocks = []
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        for model in self.base_models_:
            base_pred_blocks.append(model.predict_proba(X_aligned))
        meta_features = np.column_stack(base_pred_blocks)
        return self.meta_model_.predict_proba(meta_features)[:, 1]
