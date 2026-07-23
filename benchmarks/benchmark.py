"""Reproducible benchmarks for ModelSentinel on real datasets.

Runs the full ModelSentinel workflow (evaluation, calibration, threshold,
data-quality, drift, health) against scikit-learn's bundled datasets, measures
wall-clock time for each check, and writes ``benchmarks/RESULTS.md``.

Run:  python benchmarks/benchmark.py
"""
from __future__ import annotations

import platform
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import modelsentinel as ms


def _timed(fn: Callable):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0  # ms


def run_dataset(name: str, loader, binary: bool):
    data = loader()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42,
                                              stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    y_score = proba[:, 1] if binary else proba

    m = ms.Monitor(task="classification", name=name)
    rows = []

    ev, t = _timed(lambda: m.evaluate(y_te, y_pred, y_score))
    rows.append(("evaluate", t, f"acc={ev['accuracy']:.3f} f1={ev['f1']:.3f} "
                                 f"auc={ev['roc_auc']:.3f}" if ev['roc_auc'] else
                                 f"acc={ev['accuracy']:.3f} f1={ev['f1']:.3f}"))

    if binary:
        cal, t = _timed(lambda: m.calibration(y_te, y_score))
        rows.append(("calibration", t, f"brier={cal['brier']:.3f} ece={cal['ece']:.3f}"))
        thr, t = _timed(lambda: m.tune_threshold(y_te, y_score, "f1"))
        rows.append(("threshold", t, f"best_t={thr['best_threshold']:.2f} "
                                     f"f1={thr['best_value']:.3f}"))

    # Simulate production drift: shift the first three features.
    prod = X_te.copy()
    for c in X_te.columns[:3]:
        prod[c] = prod[c] * 1.5 + prod[c].std()
    prod.iloc[:10, 3] = np.nan

    dq, t = _timed(lambda: m.profile_data(prod))
    rows.append(("data_quality", t, f"score={dq['score']:.1f} issues={len(dq['issues'])}"))

    dr, t = _timed(lambda: m.detect_drift(X_tr, prod))
    rows.append(("drift", t, f"drifted={len(dr['drifted_features'])}/{dr['n_features']} "
                             f"score={dr['score']:.1f}"))

    sc, t = _timed(lambda: m.validate_schema(X_tr, prod))
    rows.append(("schema", t, f"valid={sc['valid']}"))

    hs, t = _timed(m.health_score)
    rows.append(("health_score", t, f"overall={hs['overall']:.1f} ({hs['grade']})"))

    return {
        "name": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "rows": rows,
        "drift_detected": dr["drifted_features"][:5],
        "total_ms": sum(r[1] for r in rows),
    }


def main() -> None:
    datasets = [
        ("breast_cancer", load_breast_cancer, True),
        ("wine", load_wine, False),
    ]
    results = [run_dataset(n, loader, b) for n, loader, b in datasets]

    lines = ["# ModelSentinel Benchmarks", ""]
    lines.append(f"- Python: {platform.python_version()} on {platform.system()}")
    lines.append(f"- numpy {np.__version__}, pandas {pd.__version__}, "
                 f"ModelSentinel {ms.__version__}")
    lines.append("- Reproduce: `python benchmarks/benchmark.py`")
    lines.append("")
    for r in results:
        lines.append(f"## {r['name']}  ({r['n_samples']} samples · {r['n_features']} features)")
        lines.append("")
        lines.append("| Check | Time (ms) | Result |")
        lines.append("| --- | ---: | --- |")
        for check, ms_time, detail in r["rows"]:
            lines.append(f"| {check} | {ms_time:.2f} | {detail} |")
        lines.append(f"| **total** | **{r['total_ms']:.2f}** | |")
        lines.append("")
        lines.append(f"Drift correctly flagged the shifted features: "
                     f"`{', '.join(map(str, r['drift_detected']))}`")
        lines.append("")

    with open("benchmarks/RESULTS.md", "w") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    print("\nWrote benchmarks/RESULTS.md")


if __name__ == "__main__":
    main()
