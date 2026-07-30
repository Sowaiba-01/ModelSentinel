import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from modelsentinel.adapters import ModelAdapter, from_sklearn


def test_from_sklearn_predict_and_proba():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    adapter = from_sklearn(clf, name="logreg")
    assert adapter.name == "logreg"
    assert adapter.predict(X).shape == (200,)
    assert adapter.has_proba()
    assert adapter.predict_proba(X).shape == (200, 2)


def test_adapter_from_callables():
    adapter = ModelAdapter(predict=lambda X: np.zeros(len(X)), name="dummy")
    assert adapter.predict([[1], [2], [3]]).tolist() == [0.0, 0.0, 0.0]
    assert adapter.has_proba() is False
    with pytest.raises(NotImplementedError):
        adapter.predict_proba([[1]])


def test_adapter_requires_model_or_predict():
    with pytest.raises(ValueError):
        ModelAdapter()
