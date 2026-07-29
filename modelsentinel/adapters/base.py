"""A uniform interface over any model.

Different frameworks expose predictions differently (``predict``,
``predict_proba``, ``__call__``, logits, ...). ``ModelAdapter`` normalizes them
so the rest of ModelSentinel can treat every model the same way.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class ModelAdapter:
    """Wrap a model (or raw callables) behind ``predict`` / ``predict_proba``.

    Parameters
    ----------
    model:
        Any object exposing ``predict`` (and optionally ``predict_proba``).
    predict:
        Optional callable used instead of ``model.predict``.
    predict_proba:
        Optional callable used instead of ``model.predict_proba``.
    name:
        Human-readable name used in reports.
    """

    def __init__(self, model=None, predict: Callable | None = None,
                 predict_proba: Callable | None = None, name: str = "model"):
        if model is None and predict is None:
            raise ValueError("provide either a model or a predict callable")
        self.model = model
        self.name = name
        self._predict = predict
        self._predict_proba = predict_proba

    def predict(self, X) -> np.ndarray:
        if self._predict is not None:
            return np.asarray(self._predict(X))
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X) -> np.ndarray:
        if self._predict_proba is not None:
            return np.asarray(self._predict_proba(X))
        if self.model is not None and hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        raise NotImplementedError(f"{self.name} does not expose predict_proba")

    def has_proba(self) -> bool:
        return self._predict_proba is not None or (
            self.model is not None and hasattr(self.model, "predict_proba")
        )
