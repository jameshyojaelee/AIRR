"""
Approximate TCRdist-inspired kNN repertoire classifier using k-mer vectors.

Applies kNN with cosine distance on an existing repertoire-level feature matrix
(typically k-mer features). This is a lightweight surrogate for full TCRdist.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from airrml.models import register_model
from airrml.models.base import BaseRepertoireModel


@register_model("tcrdist_knn")
class TCRdistKNNModel(BaseRepertoireModel):
    def __init__(self, k_neighbors: int = 5, kmer_k: int = 3, **kwargs: Any) -> None:
        super().__init__(k_neighbors=k_neighbors, kmer_k=kmer_k, **kwargs)
        self.k_neighbors = k_neighbors
        self.kmer_k = kmer_k
        self.knn_: Optional[KNeighborsClassifier] = None
        self.feature_names_: Optional[list[str]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "TCRdistKNNModel":
        self.feature_names_ = list(X_train.columns)
        X_norm = normalize(X_train, norm="l2")
        self.knn_ = KNeighborsClassifier(n_neighbors=self.k_neighbors, metric="cosine", weights="distance")
        self.knn_.fit(X_norm, y_train.loc[X_train.index])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.knn_ is None or self.feature_names_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        X_norm = normalize(X_aligned, norm="l2")
        return self.knn_.predict_proba(X_norm)[:, 1]
