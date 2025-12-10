"""
Approximate TCRdist-inspired kNN repertoire classifier using k-mer vectors.

Represents each repertoire by 3-mer TF vectors and applies KNN with cosine
similarity. This is a lightweight surrogate for full TCRdist.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from airrml.features import build_kmer_features, apply_kmer_features
from airrml.models import register_model
from airrml.models.base import BaseRepertoireModel


@register_model("tcrdist_knn")
class TCRdistKNNModel(BaseRepertoireModel):
    def __init__(self, k_neighbors: int = 5, kmer_k: int = 3, **kwargs: Any) -> None:
        super().__init__(k_neighbors=k_neighbors, kmer_k=kmer_k, **kwargs)
        self.k_neighbors = k_neighbors
        self.kmer_k = kmer_k
        self.knn_: Optional[KNeighborsClassifier] = None
        self.feature_names_: Optional[pd.Index] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "TCRdistKNNModel":
        X_kmer, _, kmer_info = build_kmer_features(X_train, k=self.kmer_k)
        X_norm = normalize(X_kmer, norm="l2")
        self.feature_info = {"kmer": kmer_info}
        self.feature_names_ = X_norm.columns
        dist_metric = "cosine"
        self.knn_ = KNeighborsClassifier(n_neighbors=self.k_neighbors, metric=dist_metric, weights="distance")
        self.knn_.fit(X_norm, y_train.loc[X_kmer.index])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.knn_ is None or self.feature_info is None:
            raise RuntimeError("Model has not been fitted.")
        X_kmer = apply_kmer_features(X, self.feature_info["kmer"])
        X_kmer = X_kmer.reindex(columns=self.feature_names_, fill_value=0)
        X_norm = normalize(X_kmer, norm="l2")
        return self.knn_.predict_proba(X_norm)[:, 1]
