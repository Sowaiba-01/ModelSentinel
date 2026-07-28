"""Feature effect curves (1-D partial dependence)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def feature_effect(
    predict_fn: Callable,
    X,
    feature: int,
    grid: int = 20,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> dict[str, object]:
    """How the average prediction changes as one feature is swept.

    This is a model-agnostic 1-D partial-dependence curve: fix every row's
    ``feature`` to a grid value, average the model's output, and repeat across
    the grid. A flat curve means the feature barely matters; a steep one means
    it drives predictions.
    """
    X = np.asarray(X, dtype=float)
    col = X[:, feature]
    values = np.linspace(np.percentile(col, lower_pct), np.percentile(col, upper_pct), grid)

    avg_prediction = np.empty(grid, dtype=float)
    for i, v in enumerate(values):
        Xp = X.copy()
        Xp[:, feature] = v
        avg_prediction[i] = float(np.mean(np.asarray(predict_fn(Xp), dtype=float)))

    span = float(avg_prediction.max() - avg_prediction.min())
    return {
        "feature": feature,
        "values": values.tolist(),
        "avg_prediction": avg_prediction.tolist(),
        "effect_span": span,
    }
