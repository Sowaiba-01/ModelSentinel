"""Aggregate sub-scores into a single Model Health Score."""
from __future__ import annotations

DEFAULT_WEIGHTS = {
    "performance": 0.35,
    "data_quality": 0.25,
    "drift": 0.25,
    "reliability": 0.15,
}


def _grade(score: float) -> str:
    if score >= 90:
        return "EXCELLENT"
    if score >= 75:
        return "GOOD"
    if score >= 60:
        return "WARNING"
    return "CRITICAL"


def health_score(
    performance: float | None = None,
    data_quality: float | None = None,
    drift: float | None = None,
    reliability: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    """Combine available sub-scores (each 0-100) into an overall health score.

    Missing components are simply ignored and the remaining weights are
    renormalised, so the score is meaningful even with partial information.
    """
    weights = weights or DEFAULT_WEIGHTS
    components = {
        "performance": performance,
        "data_quality": data_quality,
        "drift": drift,
        "reliability": reliability,
    }
    present = {k: v for k, v in components.items() if v is not None}
    if not present:
        raise ValueError("health_score needs at least one sub-score")

    total_w = sum(weights.get(k, 0.0) for k in present) or 1.0
    overall = sum(present[k] * weights.get(k, 0.0) for k in present) / total_w

    breakdown = {k: {"score": round(float(v), 2), "grade": _grade(v)}
                 for k, v in present.items()}
    return {
        "overall": round(float(overall), 2),
        "grade": _grade(overall),
        "components": breakdown,
    }
