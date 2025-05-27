import pickle
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression

from my_trend_app.model import TrendModel


def test_predict_probability():
    X = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    y = np.array([0, 1])
    model = LogisticRegression().fit(X, y)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name

    tm = TrendModel(tmp_path)
    features = {
        '1H': {'ADX': 1, '%Above20MA': 2, 'R2': 3},
        '4H': {'ADX': 4, '%Above20MA': 5, 'R2': 6},
    }
    prob = tm.predict_probability(features)
    assert 0.0 <= prob <= 1.0
