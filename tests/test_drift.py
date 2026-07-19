from modelsentinel.monitoring import detect_drift, validate_schema


def test_no_drift_same_distribution(reference_df):
    res = detect_drift(reference_df, reference_df.copy())
    assert res["dataset_drift"] is False
    assert res["score"] == 100.0


def test_drift_detected(reference_df, drifted_df):
    res = detect_drift(reference_df, drifted_df)
    assert res["dataset_drift"] is True
    assert "num" in res["drifted_features"]
    assert res["score"] < 100.0
    assert res["features"]["num"]["psi"] >= 0


def test_schema_valid(reference_df):
    res = validate_schema(reference_df, reference_df.copy())
    assert res["valid"] is True


def test_schema_mismatch(reference_df):
    modified = reference_df.drop(columns=["cat"]).assign(extra=1)
    res = validate_schema(reference_df, modified)
    assert res["valid"] is False
    assert "cat" in res["missing_columns"]
    assert "extra" in res["unexpected_columns"]
