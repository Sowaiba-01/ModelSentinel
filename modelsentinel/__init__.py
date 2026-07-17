"""ModelSentinel — AI reliability & observability toolkit.

Monitor, evaluate, explain, and protect machine-learning models with a single,
consistent Python API.

Quick start
-----------
>>> from modelsentinel import Monitor
>>> monitor = Monitor(task="classification")
>>> monitor.evaluate(y_true, y_pred, y_score)
>>> monitor.detect_drift(reference_df, production_df)
>>> monitor.profile_data(production_df)
>>> monitor.health_score()
>>> monitor.generate_report("report.html")
"""

from modelsentinel.core.config import MonitorConfig
from modelsentinel.core.monitor import Monitor
from modelsentinel.data_quality import profile_data
from modelsentinel.evaluation import (
    calibration_report,
    evaluate_classification,
    evaluate_regression,
    optimal_threshold,
)
from modelsentinel.health import health_score
from modelsentinel.monitoring import detect_drift, validate_schema

__version__ = "0.3.0"

__all__ = [
    "Monitor",
    "MonitorConfig",
    "evaluate_classification",
    "evaluate_regression",
    "calibration_report",
    "optimal_threshold",
    "profile_data",
    "detect_drift",
    "validate_schema",
    "health_score",
    "__version__",
]
