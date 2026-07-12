"""Classification evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(
    y_true,
    y_pred,
    y_score=None,
    average: str = "weighted",
) -> dict[str, object]:
    """Compute a standard suite of classification metrics.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted labels.
    y_score:
        Optional predicted probabilities/scores. For binary problems this is the
        probability of the positive class; enables ROC-AUC.
    average:
        Averaging strategy passed to precision/recall/F1 for multiclass targets.

    Returns
    -------
    dict
        Metrics plus a confusion matrix and a bounded ``score`` in [0, 100].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    labels: list = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    metrics["labels"] = labels
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    roc_auc: float | None = None
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if y_score.ndim == 1 or y_score.shape[1] == 1:
                roc_auc = float(roc_auc_score(y_true, np.ravel(y_score)))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average=average)
                )
        except (ValueError, IndexError):
            roc_auc = None
    metrics["roc_auc"] = roc_auc

    # A single bounded quality score: F1 is the backbone, nudged by ROC-AUC.
    score = metrics["f1"]
    if roc_auc is not None:
        score = 0.7 * metrics["f1"] + 0.3 * roc_auc
    metrics["score"] = round(float(score) * 100, 2)
    return metrics
