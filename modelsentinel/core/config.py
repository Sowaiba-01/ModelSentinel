"""Configuration objects for ModelSentinel."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonitorConfig:
    """Tunable thresholds and weights for a :class:`~modelsentinel.Monitor`.

    Attributes
    ----------
    task:
        Either ``"classification"`` or ``"regression"``.
    drift_psi_threshold:
        PSI above this value flags a feature as drifted (0.2 is a common rule).
    drift_pvalue_threshold:
        p-value below this flags a statistically significant distribution shift.
    outlier_iqr_multiplier:
        Multiplier applied to the IQR when screening numeric outliers.
    health_weights:
        Relative weights used to combine sub-scores into an overall health score.
    """

    task: str = "classification"
    drift_psi_threshold: float = 0.2
    drift_pvalue_threshold: float = 0.05
    outlier_iqr_multiplier: float = 1.5
    health_weights: dict[str, float] = field(
        default_factory=lambda: {
            "performance": 0.35,
            "data_quality": 0.25,
            "drift": 0.25,
            "reliability": 0.15,
        }
    )

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError(
                f"task must be 'classification' or 'regression', got {self.task!r}"
            )
