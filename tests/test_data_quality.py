import numpy as np
import pandas as pd

from modelsentinel.data_quality import profile_data


def test_clean_data_high_score():
    df = pd.DataFrame({"a": range(100), "b": np.random.default_rng(0).normal(0, 1, 100)})
    res = profile_data(df)
    assert res["score"] > 95
    assert res["missing_total"] == 0
    assert res["duplicate_rows"] == 0


def test_detects_issues():
    df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1.0, np.nan, 3.0, 3.0], "c": [5, 5, 5, 5]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add a duplicate
    res = profile_data(df)
    assert res["missing_total"] >= 1
    assert res["duplicate_rows"] >= 1
    assert set(res["constant_columns"]) >= {"a", "c"}
    assert res["issues"]
    assert res["score"] < 100
