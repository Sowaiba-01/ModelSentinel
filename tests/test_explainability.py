import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from modelsentinel.explainability import feature_effect, grad_cam, permutation_importance


def _fitted_model():
    X, y = make_classification(n_samples=500, n_features=6, n_informative=3,
                               n_redundant=0, shuffle=False, random_state=0)
    clf = RandomForestClassifier(n_estimators=120, random_state=0).fit(X, y)
    return clf, X, y


def test_permutation_importance_ranks_informative_features():
    clf, X, y = _fitted_model()
    res = permutation_importance(clf.predict, X, y, n_repeats=5)
    assert 0.0 <= res["baseline"] <= 1.0
    assert len(res["ranking"]) == 6
    # informative features are the first three (shuffle=False); at least one
    # should out-rank the least important feature.
    informative = {"f0", "f1", "f2"}
    top3 = set(res["ranking"][:3])
    assert informative & top3


def test_permutation_importance_custom_names():
    clf, X, y = _fitted_model()
    names = [f"col{i}" for i in range(6)]
    res = permutation_importance(clf.predict, X, y, n_repeats=2, feature_names=names)
    assert set(res["importances"]) == set(names)


def test_feature_effect_curve():
    clf, X, y = _fitted_model()
    res = feature_effect(clf.predict, X, feature=0, grid=15)
    assert len(res["values"]) == 15
    assert len(res["avg_prediction"]) == 15
    assert res["effect_span"] >= 0.0


def test_grad_cam_shapes_and_peak():
    acts = np.zeros((8, 5, 5))
    acts[:, 2, 3] = 1.0            # signal at (2,3)
    grads = np.ones((8, 5, 5))
    cam = grad_cam(acts, grads)
    assert cam.shape == (5, 5)
    assert cam.min() >= 0.0 and cam.max() == pytest.approx(1.0)
    assert np.unravel_index(cam.argmax(), cam.shape) == (2, 3)


def test_grad_cam_accepts_batch_dim():
    acts = np.random.default_rng(0).random((1, 4, 6, 6))
    grads = np.random.default_rng(1).random((1, 4, 6, 6))
    cam = grad_cam(acts, grads)
    assert cam.shape == (6, 6)
