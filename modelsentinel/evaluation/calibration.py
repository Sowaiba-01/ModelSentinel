"""Probability calibration diagnostics for binary classifiers."""
from __future__ import annotations

import numpy as np


def calibration_report(y_true, y_score, n_bins: int = 10) -> dict[str, object]:
    """Assess how well predicted probabilities match observed frequencies.

    Returns the Brier score, Expected Calibration Error (ECE), Maximum
    Calibration Error (MCE), and per-bin reliability data suitable for plotting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    brier = float(np.mean((y_score - y_true) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score > lo) & (y_score <= hi) if i > 0 else (y_score >= lo) & (y_score <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lower": float(lo), "upper": float(hi), "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        gap = abs(accuracy - confidence)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append({"lower": float(lo), "upper": float(hi), "count": count,
                     "confidence": confidence, "accuracy": accuracy})

    return {
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins,
        "score": round(max(0.0, 1.0 - ece) * 100, 2),
    }
