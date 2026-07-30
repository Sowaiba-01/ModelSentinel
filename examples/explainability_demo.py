"""Explainability demo (v0.4): permutation importance, feature effects, Grad-CAM.

Run:  python examples/explainability_demo.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import modelsentinel as ms


def main() -> None:
    X, y = make_classification(n_samples=1500, n_features=8, n_informative=4,
                               random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42).fit(X, y)

    monitor = ms.Monitor(model=clf, task="classification", name="rf-explain")

    print("== Permutation importance (top 5) ==")
    imp = monitor.explain_importance(X, y, n_repeats=5)
    for name in imp["ranking"][:5]:
        stat = imp["importances"][name]
        print(f"  {name}: {stat['importance_mean']:.4f} +/- {stat['importance_std']:.4f}")

    print("\n== Feature effect for the top feature ==")
    top = int(imp["ranking"][0][1:])  # 'f3' -> 3
    eff = monitor.feature_effect(X, top, grid=10)
    print(f"  feature f{top} effect span: {eff['effect_span']:.4f}")

    print("\n== Grad-CAM (framework-agnostic) ==")
    # In practice you'd pull these from your CNN's target layer; here we fake them.
    acts = np.zeros((16, 7, 7))
    acts[:, 3, 4] = 1.0
    grads = np.ones((16, 7, 7))
    cam = ms.grad_cam(acts, grads)
    print(f"  heatmap {cam.shape}, peak at {np.unravel_index(cam.argmax(), cam.shape)}")


if __name__ == "__main__":
    main()
