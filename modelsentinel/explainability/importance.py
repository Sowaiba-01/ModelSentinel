"""Model-agnostic permutation feature importance."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np


def permutation_importance(
    predict_fn: Callable,
    X,
    y,
    scorer: Callable | None = None,
    n_repeats: int = 5,
    feature_names: Sequence[str] | None = None,
    random_state: int = 0,
) -> dict[str, object]:
    """Rank features by how much shuffling each one hurts model performance.

    Works with *any* model: you pass a ``predict_fn`` that maps ``X`` to
    predictions in the same space your ``scorer`` expects. Higher importance
    means the model relies on that feature more.

    Parameters
    ----------
    predict_fn:
        Callable mapping a 2-D array to predictions.
    X, y:
        Feature matrix and ground-truth targets.
    scorer:
        ``scorer(y_true, y_pred) -> float`` where higher is better. Defaults to
        accuracy for integer/binary targets, else R^2.
    n_repeats:
        How many times to shuffle each feature (importance is averaged).
    feature_names:
        Optional names; defaults to ``f0, f1, ...``.

    Returns
    -------
    dict
        ``baseline`` score, per-feature ``importances`` (mean + std), and a
        ``ranking`` of feature names from most to least important.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    if scorer is None:
        from sklearn.metrics import accuracy_score, r2_score
        is_clf = y.dtype.kind in "iub" or set(np.unique(y).tolist()) <= {0, 1}
        scorer = accuracy_score if is_clf else r2_score

    n_features = X.shape[1]
    names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length must match number of columns")

    baseline = float(scorer(y, predict_fn(X)))
    importances: dict[str, dict[str, float]] = {}
    for j in range(n_features):
        drops = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r] = baseline - float(scorer(y, predict_fn(Xp)))
        importances[names[j]] = {
            "importance_mean": float(drops.mean()),
            "importance_std": float(drops.std()),
        }

    ranking: list[str] = sorted(importances, key=lambda k: importances[k]["importance_mean"], reverse=True)
    return {"baseline": baseline, "importances": importances, "ranking": ranking}
