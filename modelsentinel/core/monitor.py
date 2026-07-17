"""The Monitor facade — one object that orchestrates the whole toolkit."""
from __future__ import annotations

import pandas as pd

from modelsentinel.core.config import MonitorConfig
from modelsentinel.data_quality import profile_data
from modelsentinel.evaluation import (
    calibration_report,
    evaluate_classification,
    evaluate_regression,
    optimal_threshold,
)
from modelsentinel.health import health_score
from modelsentinel.monitoring import detect_drift, validate_schema
from modelsentinel.reporting import generate_html_report


class Monitor:
    """High-level entry point for evaluating and monitoring a model.

    The Monitor accumulates results from each check into ``self.results`` so a
    single call to :meth:`health_score` or :meth:`generate_report` can summarise
    everything that has been run.

    Parameters
    ----------
    model:
        Optional model object (kept for reference/reporting only).
    task:
        ``"classification"`` or ``"regression"``.
    name:
        Human-readable model name used in reports.
    config:
        Optional :class:`MonitorConfig` overriding default thresholds/weights.
    """

    def __init__(self, model=None, task: str = "classification",
                 name: str = "model", config: MonitorConfig | None = None):
        self.model = model
        self.name = name
        self.config = config or MonitorConfig(task=task)
        self.task = self.config.task
        self.results: dict[str, object] = {}

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, y_true, y_pred, y_score=None) -> dict:
        """Evaluate predictions and cache the metrics."""
        if self.task == "classification":
            res = evaluate_classification(y_true, y_pred, y_score)
        else:
            res = evaluate_regression(y_true, y_pred)
        self.results["evaluation"] = res
        return res

    def calibration(self, y_true, y_score, n_bins: int = 10) -> dict:
        res = calibration_report(y_true, y_score, n_bins=n_bins)
        self.results["calibration"] = res
        return res

    def tune_threshold(self, y_true, y_score, metric: str = "f1") -> dict:
        res = optimal_threshold(y_true, y_score, metric=metric)
        self.results["threshold"] = res
        return res

    # -- data quality & drift ----------------------------------------------
    def profile_data(self, df: pd.DataFrame) -> dict:
        res = profile_data(df, self.config.outlier_iqr_multiplier)
        self.results["data_quality"] = res
        return res

    def detect_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        res = detect_drift(reference, current,
                           psi_threshold=self.config.drift_psi_threshold,
                           pvalue_threshold=self.config.drift_pvalue_threshold)
        self.results["drift"] = res
        return res

    def validate_schema(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        res = validate_schema(reference, current)
        self.results["schema"] = res
        return res

    # -- aggregate ----------------------------------------------------------
    def health_score(self) -> dict:
        """Combine whatever checks have been run into an overall health score."""
        ev = self.results.get("evaluation") or {}
        dq = self.results.get("data_quality") or {}
        dr = self.results.get("drift") or {}
        cal = self.results.get("calibration") or {}
        res = health_score(
            performance=ev.get("score"),
            data_quality=dq.get("score"),
            drift=dr.get("score"),
            reliability=cal.get("score"),
            weights=self.config.health_weights,
        )
        self.results["health"] = res
        return res

    def generate_report(self, output_path: str = "model_report.html") -> str:
        """Render an HTML report of all cached results."""
        if "health" not in self.results and self.results:
            try:
                self.health_score()
            except ValueError:
                pass
        return generate_html_report(
            output_path,
            model_name=self.name,
            health=self.results.get("health"),
            evaluation=self.results.get("evaluation"),
            data_quality=self.results.get("data_quality"),
            drift=self.results.get("drift"),
            schema=self.results.get("schema"),
        )
