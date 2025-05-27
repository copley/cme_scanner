from __future__ import annotations

import pickle
from typing import Any, Optional

import numpy as np


class TrendModel:
    """Wrapper around a pickled logistic regression model."""

    def __init__(self, model_path: str, logger: Optional[Any] = None) -> None:
        self.logger = logger
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_probability(self, features: dict) -> float:
        """Predict probability that trend continues."""
        timeframe_order = sorted(features.keys())
        feat_list: list[float] = []
        for tf in timeframe_order:
            tf_vals = features[tf]
            feat_list.append(tf_vals['ADX'])
            feat_list.append(tf_vals['%Above20MA'])
            feat_list.append(tf_vals['R2'])
        X = np.array(feat_list).reshape(1, -1)
        prob = self.model.predict_proba(X)[0, 1]
        return float(prob)
