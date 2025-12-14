"""
Abstract interfaces for repertoire-level models used in AIRR-ML-25.

Concrete models (classical ML or deep sequence/repertoire models) should
subclass BaseRepertoireModel and implement the abstract methods defined here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseRepertoireModel(ABC):
    """
    Base interface for repertoire-level classifiers.

    The interface is intentionally minimal and agnostic to the underlying
    modeling approach (logistic regression, gradient boosting, deep MIL, etc.).
    """
    consumes_sequences: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.name: str = self.__class__.__name__
        self.config: Dict[str, Any] = kwargs
        self.feature_info: Optional[Dict[str, Any]] = None

    def set_feature_info(self, feature_info: Dict[str, Any]) -> None:
        """
        Attach metadata about the features used (e.g., vocabularies) to enable
        consistent transforms at inference time.
        """
        self.feature_info = feature_info

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseRepertoireModel":
        """
        Train the model on repertoire-level features.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict positive-class probabilities for each repertoire.
        """
        raise NotImplementedError

    def get_repertoire_importance(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Optional diagnostic scores at the repertoire level (e.g., feature importances).
        Default returns None.
        """
        return None

    def get_sequence_importance(
        self,
        sequences_df: pd.DataFrame,
        sequence_col: str = "junction_aa",
    ) -> Optional[pd.DataFrame]:
        """
        Score individual sequences for their contribution to the positive class.

        Expected return format:
            DataFrame with columns [junction_aa, v_call, j_call, importance_score]
        The default implementation should be overridden by models that support
        sequence-level attribution.
        """
        raise NotImplementedError("Sequence-level importance is not implemented for this model.")
