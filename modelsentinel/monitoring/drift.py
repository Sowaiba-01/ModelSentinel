"""Advanced data-drift detection for tabular data.

Numeric features are compared with the two-sample Kolmogorov-Smirnov test and
the Population Stability Index (PSI). Categorical features are compared with a
chi-square test and the Jensen-Shannon divergence. Feature-level results are
aggregated into an overall drift verdict and a bounded score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from the reference sample."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _js_divergence(expected: pd.Series, actual: pd.Series) -> float:
    """Jensen-Shannon divergence between two categorical distributions."""
    cats = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    d = jensenshannon(e, a, base=2)  # 0..1 with base 2
    return float(d if not np.isnan(d) else 0.0)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
    pvalue_threshold: float = 0.05,
    columns: list[str] | None = None,
) -> dict[str, object]:
    """Detect distribution drift between a reference and current dataset.

    Returns per-feature statistics, the list of drifted features, an overall
    boolean verdict, and a bounded ``score`` in [0, 100] (100 = no drift).
    """
    cols = columns or [c for c in reference.columns if c in current.columns]
    features: dict[str, dict[str, object]] = {}
    drifted: list[str] = []

    for c in cols:
        ref, cur = reference[c].dropna(), current[c].dropna()
        if ref.empty or cur.empty:
            continue
        if pd.api.types.is_numeric_dtype(reference[c]):
            ks_stat, p_value = stats.ks_2samp(ref, cur)
            psi = _psi(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float))
            is_drift = bool(psi >= psi_threshold or p_value < pvalue_threshold)
            features[c] = {"type": "numeric", "ks_stat": float(ks_stat),
                           "p_value": float(p_value), "psi": round(psi, 4), "drift": is_drift}
        else:
            cats = sorted(set(ref.unique()) | set(cur.unique()))
            e = ref.value_counts().reindex(cats, fill_value=0).to_numpy()
            a = cur.value_counts().reindex(cats, fill_value=0).to_numpy()
            table = np.vstack([e, a])
            keep = table.sum(axis=0) > 0
            try:
                _, p_value, _, _ = stats.chi2_contingency(table[:, keep])
            except ValueError:
                p_value = 1.0
            js = _js_divergence(ref, cur)
            is_drift = bool(p_value < pvalue_threshold or js >= 0.1)
            features[c] = {"type": "categorical", "p_value": float(p_value),
                           "js_divergence": round(js, 4), "drift": is_drift}
        if features[c]["drift"]:
            drifted.append(c)

    n = max(len(features), 1)
    drift_ratio = len(drifted) / n
    score = round(max(0.0, 1.0 - drift_ratio) * 100, 2)

    return {
        "features": features,
        "drifted_features": drifted,
        "n_features": len(features),
        "drift_share": round(drift_ratio, 4),
        "dataset_drift": bool(drifted),
        "score": score,
    }
