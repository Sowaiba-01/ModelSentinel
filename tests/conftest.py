import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_data(rng):
    n = 400
    y_true = rng.integers(0, 2, n)
    # scores correlated with truth but imperfect
    y_score = np.clip(0.35 * y_true + rng.normal(0.3, 0.2, n), 0, 1)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score


@pytest.fixture
def reference_df(rng):
    n = 500
    return pd.DataFrame({
        "num": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b", "c"], n),
    })


@pytest.fixture
def drifted_df(rng):
    n = 500
    return pd.DataFrame({
        "num": rng.normal(1.5, 1, n),  # mean shift -> drift
        "cat": rng.choice(["a", "b", "c"], n, p=[0.7, 0.2, 0.1]),
    })
