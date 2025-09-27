"""Data loading and feature preparation for the evenness analysis."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import FACTS_CONFIG, EvennessPaths, required_fact_columns

_GDPR_START = datetime(2018, 5, 25)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_wide_dataset(path: Path | str | None = None, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the cleaned wide CSV with minimal type coercion."""

    csv_path = Path(path or EvennessPaths().wide_csv)
    dtype_overrides = {"decision_id": str}
    usecols = None
    if columns is not None:
        desired = set(columns)

        def _filter(col: str) -> bool:
            if col in desired:
                return True
            return any(col.startswith(f"{prefix}_") for prefix in FACTS_CONFIG.multi_value_prefixes)

        usecols = _filter
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype_overrides)
    if "decision_date" in df.columns:
        df["decision_date"] = _ensure_datetime(df["decision_date"])
    if "fine_eur" in df.columns:
        df["fine_eur"] = pd.to_numeric(df["fine_eur"], errors="coerce")
    if "fine_log1p" in df.columns:
        df["fine_log1p"] = pd.to_numeric(df["fine_log1p"], errors="coerce")
    if "enforcement_severity_index" in df.columns:
        df["enforcement_severity_index"] = pd.to_numeric(
            df["enforcement_severity_index"], errors="coerce"
        )
    return df


def _sanitize_multiselect(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Apply guardrails to multi-select indicator columns."""

    coverage_col = f"{prefix}_coverage_status"
    status_col = f"{prefix}_status"
    conflict_col = f"{prefix}_exclusivity_conflict"
    indicator_cols = [
        c
        for c in df.columns
        if c.startswith(f"{prefix}_")
        and c
        not in {
            coverage_col,
            status_col,
            conflict_col,
            f"{prefix}_known",
            f"{prefix}_unknown",
        }
    ]
    if not indicator_cols:
        return df

    mask_discussed = pd.Series(True, index=df.index)
    if coverage_col in df:
        mask_discussed &= df[coverage_col].fillna("MISSING").eq("DISCUSSED")
    if status_col in df:
        mask_discussed &= df[status_col].fillna("MISSING").eq("DISCUSSED")
    mask_conflict = pd.Series(False, index=df.index)
    if conflict_col in df:
        mask_conflict = df[conflict_col].fillna(False).astype(bool)

    invalid = (~mask_discussed) | mask_conflict
    if invalid.any():
        df.loc[invalid, indicator_cols] = np.nan
    return df


def _derive_primary_category(df: pd.DataFrame, prefix: str, fallback_col: str | None = None) -> pd.Series:
    indicator_cols = [
        c
        for c in df.columns
        if c.startswith(f"{prefix}_")
        and not c.endswith("_coverage_status")
        and not c.endswith("_status")
        and not c.endswith("_exclusivity_conflict")
        and not c.endswith("_known")
        and not c.endswith("_unknown")
    ]
    if not indicator_cols:
        if fallback_col and fallback_col in df:
            return df[fallback_col].fillna(pd.NA)
        return pd.Series(pd.NA, index=df.index)

    values = df[indicator_cols].copy()
    for col in indicator_cols:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    def collapse(row: pd.Series) -> str | pd.NA:
        active = [col.split(f"{prefix}_", 1)[1] for col in indicator_cols if row.get(col, 0) == 1]
        if not active:
            return pd.NA
        return "|".join(sorted(active))

    series = values.apply(collapse, axis=1).astype("string")
    if fallback_col and fallback_col in df:
        series = series.fillna(df[fallback_col])
    return series.astype("string")


def _normalise_categorical(
    series: pd.Series | None, index: pd.Index, default: str = "UNKNOWN"
) -> pd.Series:
    if series is None:
        return pd.Series([default] * len(index), index=index, dtype="string")
    filled = series.fillna(default).replace("", default)
    return filled.astype("string")


def build_fact_matrix(path: Path | str | None = None, discussed_only: bool = False) -> pd.DataFrame:
    """Create the facts-only matrix used across the analysis steps."""

    base_cols = required_fact_columns().union({
        "raw_q8",
        "raw_q10",
        "raw_q15",
        "decision_date",
        "n_principles_discussed",
    })
    df = load_wide_dataset(path, columns=base_cols)

    # Derive decision year/quarter if missing
    if "decision_date" in df.columns and df["decision_date"].notna().any():
        if pd.api.types.is_datetime64tz_dtype(df["decision_date"]):
            df["decision_date"] = df["decision_date"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["decision_year"] = df["decision_year"].fillna(df["decision_date"].dt.year)
        df["decision_quarter"] = df["decision_quarter"].fillna(df["decision_date"].dt.quarter)
        df["days_since_gdpr"] = (df["decision_date"] - _GDPR_START).dt.days
    else:
        df["days_since_gdpr"] = np.nan

    df["days_since_gdpr"] = pd.to_numeric(df["days_since_gdpr"], errors="coerce")

    # Apply guardrails
    for prefix in FACTS_CONFIG.guardrail_prefixes:
        if any(col.startswith(f"{prefix}_") for col in df.columns):
            df = _sanitize_multiselect(df, prefix)

    df["organization_size_tier"] = _derive_primary_category(df, "q10_org_class", "raw_q10")
    df["organization_type"] = _normalise_categorical(df.get("raw_q8"), df.index)
    df["case_origin"] = _normalise_categorical(df.get("raw_q15"), df.index)

    df["decision_year_bucket"] = pd.cut(
        df["decision_year"].fillna(-1),
        bins=[2017, 2019, 2021, 2023, 2026],
        labels=["2018-2019", "2020-2021", "2022-2023", "2024-2025"],
        right=True,
    ).astype(str)

    # Derived helper columns
    df["country_year"] = df["country_code"].astype(str) + "_" + df["decision_year"].fillna(-1).astype(int).astype(str)

    if discussed_only:
        for prefix in FACTS_CONFIG.multi_value_prefixes:
            status_col = f"{prefix}_status"
            if status_col in df:
                mask = df[status_col].fillna("MISSING").eq("DISCUSSED")
                df = df.loc[mask]

    # Ensure multi-select indicator columns are numeric Int64
    for prefix in FACTS_CONFIG.multi_value_prefixes:
        indicator_cols = [
            c
            for c in df.columns
            if c.startswith(f"{prefix}_")
            and c
            not in {
                f"{prefix}_coverage_status",
                f"{prefix}_status",
                f"{prefix}_exclusivity_conflict",
                f"{prefix}_known",
                f"{prefix}_unknown",
            }
        ]
        for col in indicator_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["breach_case"] = _normalise_categorical(df.get("breach_case"), df.index)

    numeric_fill_defaults = {
        "n_principles_discussed": 0,
        "n_principles_violated": 0,
        "n_corrective_measures": 0,
    }
    for col, default in numeric_fill_defaults.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    return df


__all__ = ["build_fact_matrix", "load_wide_dataset"]
