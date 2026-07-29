"""Adapter helper for scikit-learn estimators."""
from __future__ import annotations

from modelsentinel.adapters.base import ModelAdapter


def from_sklearn(estimator, name: str = "sklearn-model") -> ModelAdapter:
    """Wrap a fitted scikit-learn estimator in a :class:`ModelAdapter`."""
    if not hasattr(estimator, "predict"):
        raise TypeError("estimator must implement .predict()")
    return ModelAdapter(model=estimator, name=name)
