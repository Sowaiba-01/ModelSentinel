"""End-to-end ModelSentinel demo on a synthetic classification problem.

Run:  python examples/quickstart.py
Produces: model_report.html
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import modelsentinel as ms


def main() -> None:
    X, y = make_classification(n_samples=2000, n_features=8, n_informative=5,
                               random_state=42)
    cols = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    clf = RandomForestClassifier(n_estimators=120, random_state=42).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    monitor = ms.Monitor(task="classification", name="RandomForest-demo")

    print("== Evaluation ==")
    ev = monitor.evaluate(y_test, y_pred, y_score)
    print({k: ev[k] for k in ["accuracy", "f1", "roc_auc", "score"]})

    print("\n== Calibration ==")
    cal = monitor.calibration(y_test, y_score)
    print({k: cal[k] for k in ["brier", "ece", "score"]})

    print("\n== Threshold ==")
    thr = monitor.tune_threshold(y_test, y_score, metric="f1")
    print(f"best_threshold={thr['best_threshold']:.2f}  f1={thr['best_value']:.4f}  "
          f"(vs default 0.5)")

    # Simulate production drift by shifting the test features.
    production = X_test.copy()
    production["f0"] = production["f0"] + 2.0
    production.iloc[:15, 1] = np.nan

    print("\n== Data quality (production) ==")
    dq = monitor.profile_data(production)
    print("score:", dq["score"], "| issues:", dq["issues"])

    print("\n== Drift (train vs production) ==")
    dr = monitor.detect_drift(X_train, production)
    print("drifted:", dr["drifted_features"], "| score:", dr["score"])

    print("\n== Schema ==")
    print(monitor.validate_schema(X_train, production))

    print("\n== Health score ==")
    print(monitor.health_score())

    path = monitor.generate_report("model_report.html")
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
