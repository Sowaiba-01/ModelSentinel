"""Data-quality profiling for tabular datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile_data(df: pd.DataFrame, outlier_iqr_multiplier: float = 1.5) -> dict[str, object]:
    """Profile a dataframe for common data-quality problems.

    Detects missing values, duplicate rows, constant columns, and IQR-based
    numeric outliers, and returns a bounded ``score`` in [0, 100] summarising
    overall quality.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile_data expects a pandas DataFrame")

    n_rows, n_cols = df.shape
    cells = max(n_rows * n_cols, 1)

    missing_by_col = df.isna().sum()
    total_missing = int(missing_by_col.sum())
    missing_pct = {c: round(float(missing_by_col[c]) / max(n_rows, 1) * 100, 4)
                   for c in df.columns}

    duplicate_rows = int(df.duplicated().sum())
    constant_cols: list[str] = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    outliers: dict[str, dict[str, float]] = {}
    total_outliers = 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo = q1 - outlier_iqr_multiplier * iqr
        hi = q3 + outlier_iqr_multiplier * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out:
            outliers[c] = {"count": n_out, "lower": float(lo), "upper": float(hi),
                           "pct": round(n_out / max(n_rows, 1) * 100, 4)}
            total_outliers += n_out

    schema = {c: str(df[c].dtype) for c in df.columns}

    # Score: start at 100 and subtract bounded penalties for each issue class.
    missing_penalty = min(40.0, total_missing / cells * 100.0)
    dup_penalty = min(20.0, duplicate_rows / max(n_rows, 1) * 100.0)
    const_penalty = min(20.0, len(constant_cols) / max(n_cols, 1) * 100.0)
    outlier_penalty = min(20.0, total_outliers / cells * 100.0)
    score = max(0.0, 100.0 - missing_penalty - dup_penalty - const_penalty - outlier_penalty)

    issues: list[str] = []
    if total_missing:
        issues.append(f"{total_missing} missing values across {int((missing_by_col > 0).sum())} column(s)")
    if duplicate_rows:
        issues.append(f"{duplicate_rows} duplicate row(s)")
    if constant_cols:
        issues.append(f"constant column(s): {', '.join(constant_cols)}")
    if outliers:
        issues.append(f"{total_outliers} outlier(s) across {len(outliers)} column(s)")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "schema": schema,
        "missing_total": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_cols,
        "outliers": outliers,
        "issues": issues,
        "score": round(score, 2),
    }
