"""Regression evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Compute standard regression metrics plus a bounded quality score."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom))) if denom.size else float("nan")
    metrics["mape"] = mape

    # Map R^2 (which can be negative) into a bounded 0-100 score.
    metrics["score"] = round(max(0.0, min(1.0, metrics["r2"])) * 100, 2)
    return metrics
