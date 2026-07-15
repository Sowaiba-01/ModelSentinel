"""Schema validation between a reference (training) and current dataset."""
from __future__ import annotations

import pandas as pd


def validate_schema(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    """Compare column sets and dtypes between two dataframes.

    Flags missing columns, unexpected new columns, and dtype mismatches — the
    silent failures that break production models before accuracy ever moves.
    """
    ref_cols = list(reference.columns)
    cur_cols = list(current.columns)

    missing: list[str] = [c for c in ref_cols if c not in cur_cols]
    unexpected: list[str] = [c for c in cur_cols if c not in ref_cols]

    dtype_mismatch: dict[str, dict[str, str]] = {}
    for c in ref_cols:
        if c in cur_cols:
            rt, ct = str(reference[c].dtype), str(current[c].dtype)
            if rt != ct:
                dtype_mismatch[c] = {"reference": rt, "current": ct}

    valid = not missing and not unexpected and not dtype_mismatch
    return {
        "valid": valid,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "dtype_mismatch": dtype_mismatch,
    }
