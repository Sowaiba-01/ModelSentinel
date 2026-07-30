import os

from modelsentinel import Monitor, __version__, health_score


def test_version():
    assert __version__ == "0.4.0"


def test_health_score_partial():
    res = health_score(performance=90, data_quality=80)
    assert 80 <= res["overall"] <= 90
    assert res["grade"] in {"EXCELLENT", "GOOD", "WARNING", "CRITICAL"}
    assert set(res["components"]) == {"performance", "data_quality"}


def test_monitor_end_to_end(binary_data, reference_df, drifted_df, tmp_path):
    y_true, y_pred, y_score = binary_data
    m = Monitor(task="classification", name="test-model")
    m.evaluate(y_true, y_pred, y_score)
    m.calibration(y_true, y_score)
    m.profile_data(drifted_df)
    m.detect_drift(reference_df, drifted_df)
    m.validate_schema(reference_df, drifted_df)
    health = m.health_score()
    assert 0 <= health["overall"] <= 100

    out = tmp_path / "report.html"
    path = m.generate_report(str(out))
    assert os.path.exists(path)
    content = out.read_text(encoding="utf-8")
    assert "ModelSentinel" in content and "test-model" in content
