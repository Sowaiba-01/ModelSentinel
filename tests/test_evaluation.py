import numpy as np

from modelsentinel.evaluation import (
    calibration_report,
    evaluate_classification,
    evaluate_regression,
    optimal_threshold,
)


def test_classification_metrics(binary_data):
    y_true, y_pred, y_score = binary_data
    res = evaluate_classification(y_true, y_pred, y_score)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "score"]:
        assert k in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["score"] <= 100.0
    assert len(res["confusion_matrix"]) == len(res["labels"])


def test_classification_without_scores(binary_data):
    y_true, y_pred, _ = binary_data
    res = evaluate_classification(y_true, y_pred)
    assert res["roc_auc"] is None
    assert 0.0 <= res["score"] <= 100.0


def test_regression_metrics():
    y_true = np.arange(100, dtype=float)
    y_pred = y_true + np.random.default_rng(1).normal(0, 1, 100)
    res = evaluate_regression(y_true, y_pred)
    assert res["r2"] > 0.9
    assert res["mae"] >= 0
    assert 0.0 <= res["score"] <= 100.0


def test_calibration(binary_data):
    y_true, _, y_score = binary_data
    res = calibration_report(y_true, y_score, n_bins=10)
    assert 0.0 <= res["ece"] <= 1.0
    assert res["brier"] >= 0.0
    assert len(res["bins"]) == 10


def test_optimal_threshold(binary_data):
    y_true, _, y_score = binary_data
    res = optimal_threshold(y_true, y_score, metric="f1")
    assert 0.0 <= res["best_threshold"] <= 1.0
    yj = optimal_threshold(y_true, y_score, metric="youden")
    assert 0.0 <= yj["best_threshold"] <= 1.0
