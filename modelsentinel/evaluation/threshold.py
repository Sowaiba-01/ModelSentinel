"""Decision-threshold analysis for binary classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def optimal_threshold(y_true, y_score, metric: str = "f1", n_steps: int = 101) -> dict[str, object]:
    """Sweep decision thresholds and report the one that maximises ``metric``.

    Supported metrics: ``"f1"`` and ``"youden"`` (Youden's J = TPR - FPR).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.0, 1.0, n_steps)

    best_t, best_v = 0.5, -np.inf
    curve = []
    P = max(int((y_true == 1).sum()), 1)
    N = max(int((y_true == 0).sum()), 1)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        if metric == "f1":
            value = f1_score(y_true, pred, zero_division=0)
        elif metric == "youden":
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            value = (tp / P) - (fp / N)
        else:
            raise ValueError("metric must be 'f1' or 'youden'")
        curve.append({"threshold": float(t), "value": float(value)})
        if value > best_v:
            best_v, best_t = float(value), float(t)

    return {"metric": metric, "best_threshold": best_t, "best_value": best_v, "curve": curve}
