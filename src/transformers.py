from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .features import FEATURES, add_features

@dataclass
class FeatureAdder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that adds engineered features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X can be a DataFrame (preferred) or ndarray
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=FEATURES)
        return add_features(df)
