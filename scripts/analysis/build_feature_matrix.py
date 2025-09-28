from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


BASE_COLUMNS: tuple[str, ...] = (
    "decision_id",
    "country_code",
    "country_group",
    "dpa_name_canonical",
    "decision_year",
    "decision_quarter",
    "breach_case",
    "severity_measures_present",
    "remedy_only_case",
    "fine_eur",
    "fine_log1p",
    "fine_status",
    "fine_to_turnover_ratio",
    "turnover_eur",
    "turnover_log1p",
    "turnover_status",
    "isic_section",
    "n_principles_discussed",
    "n_principles_violated",
    "n_corrective_measures",
)

ENUM_COLUMNS: tuple[str, ...] = (
    "art33_notification_required",
    "art33_notification_required_status",
    "art33_notification_submitted",
    "art33_notification_submitted_status",
    "art33_notification_timeliness",
    "art33_notification_timeliness_status",
    "art33_notification_delay_band",
    "art33_notification_delay_band_status",
    "data_subjects_notified",
    "data_subjects_notified_status",
    "art34_notification_required",
    "art34_notification_required_status",
)

MULTI_BASES: tuple[str, ...] = (
    "q10_org_class",
    "q15_case_initiation",
    "q21_breach_types",
    "q25_sensitive_data",
    "q28_mitigations",
    "q30_discussed",
    "q31_violated",
    "q32_bases",
    "q41_aggrav",
    "q42_mitig",
    "q46_vuln",
    "q47_remedial",
    "q50_other_measures",
    "q53_powers",
    "q54_scopes",
    "q56_rights_discussed",
    "q57_rights_violated",
    "q58_access_issues",
    "q59_adm_issues",
    "q61_dpo_issues",
    "q64_transfer_violations",
)

META_SUFFIXES: tuple[str, ...] = (
    "_coverage_status",
    "_known",
    "_unknown",
    "_status",
    "_exclusivity_conflict",
)

POWER_TOKEN_COLUMNS: tuple[str, ...] = (
    "q53_powers_WARNING",
    "q53_powers_REPRIMAND",
    "q53_powers_COMPLY_WITH_DATA_SUBJECT_REQUESTS",
    "q53_powers_BRING_PROCESSING_INTO_COMPLIANCE",
    "q53_powers_COMMUNICATE_BREACH_TO_SUBJECTS",
    "q53_powers_LIMITATION_PROHIBITION_OF_PROCESSING",
    "q53_powers_RECTIFICATION_ERASURE_RESTRICTION",
    "q53_powers_CERTIFICATION_WITHDRAWAL",
    "q53_powers_ADMINISTRATIVE_FINE",
    "q53_powers_SUSPENSION_DATA_FLOWS",
    "q53_powers_NONE",
)


@dataclass(frozen=True)
class FeatureMatrixArtifacts:
    dataframe: pd.DataFrame
    column_groups: dict[str, Sequence[str]]


def _read_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("wide dataset is empty; ensure the cleaning pipeline ran successfully")
    return df


def _collect_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "base": [col for col in BASE_COLUMNS if col in df.columns],
        "enums": [col for col in ENUM_COLUMNS if col in df.columns],
    }
    for base in MULTI_BASES:
        prefix = f"{base}_"
        matches = [col for col in df.columns if col.startswith(prefix)]
        if matches:
            groups[base] = matches
    return groups


def _convert_indicator_columns(matrix: pd.DataFrame, column_groups: dict[str, Sequence[str]]) -> None:
    for base in MULTI_BASES:
        cols = column_groups.get(base, [])
        for col in cols:
            if col.endswith(META_SUFFIXES):
                continue
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").astype("Int64")


def _derive_notification_flags(matrix: pd.DataFrame) -> None:
    def build_flag(column: str, positive_values: set[str]) -> pd.Series:
        series = matrix.get(column)
        flag = pd.Series(pd.NA, index=matrix.index, dtype="Int64")
        if series is None:
            return flag
        matches = series.isin(positive_values)
        non_matches = series.notna() & ~matches
        flag[matches] = 1
        flag[non_matches] = 0
        return flag

    matrix["art33_required_flag"] = build_flag("art33_notification_required", {"YES_REQUIRED"})
    matrix["art33_submitted_flag"] = build_flag("art33_notification_submitted", {"YES_SUBMITTED"})
    matrix["art33_late_flag"] = build_flag("art33_notification_timeliness", {"NO_LATE"})
    matrix["art33_timely_flag"] = build_flag("art33_notification_timeliness", {"YES_WITHIN_72H"})
    matrix["art34_required_flag"] = build_flag("art34_notification_required", {"YES_REQUIRED"})
    matrix["subjects_notified_flag"] = build_flag(
        "data_subjects_notified", {"YES_NOTIFIED", "PARTIALLY_NOTIFIED"}
    )


def _derive_power_features(matrix: pd.DataFrame) -> None:
    for col in POWER_TOKEN_COLUMNS:
        if col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").astype("Int64")
        else:
            matrix[col] = pd.Series(dtype="Int64")

    power_frame = matrix[list(POWER_TOKEN_COLUMNS)].fillna(0)
    matrix["power_option_count"] = power_frame.sum(axis=1).astype(int)
    matrix["power_any_flag"] = (power_frame.sum(axis=1) > 0).astype(int)
    matrix["power_fine_flag"] = power_frame["q53_powers_ADMINISTRATIVE_FINE"].astype(int)
    matrix["power_warning_flag"] = power_frame["q53_powers_WARNING"].astype(int)
    matrix["power_reprimand_flag"] = power_frame["q53_powers_REPRIMAND"].astype(int)
    matrix["power_none_flag"] = power_frame["q53_powers_NONE"].astype(int)
    matrix["power_combined_flag"] = (power_frame.sum(axis=1) > 1).astype(int)
    matrix["power_non_fine_flag"] = (
        (power_frame.drop(columns=["q53_powers_ADMINISTRATIVE_FINE", "q53_powers_NONE"], errors="ignore").sum(axis=1) > 0)
    ).astype(int)


def _pool_case_initiation(matrix: pd.DataFrame, column_groups: dict[str, list[str]], threshold: int = 10) -> None:
    columns = column_groups.get("q15_case_initiation")
    if not columns:
        return
    indicator_cols = [
        col
        for col in columns
        if not col.endswith(META_SUFFIXES)
        and col not in {"q15_case_initiation_known", "q15_case_initiation_unknown"}
        and col in matrix.columns
    ]
    low_freq = []
    for col in indicator_cols:
        values = matrix[col]
        if values.notna().any():
            count = (pd.to_numeric(values, errors="coerce") == 1).sum()
            if count < threshold:
                low_freq.append(col)
    if not low_freq:
        return
    pooled_values = pd.Series(0, index=matrix.index, dtype="Int64")
    for col in low_freq:
        pooled_values = pooled_values | (pd.to_numeric(matrix[col], errors="coerce").fillna(0).astype(int) == 1)
    pooled_column = "q15_case_initiation_LOW_FREQUENCY"
    matrix[pooled_column] = pooled_values.astype("Int64")
    for col in low_freq:
        matrix.drop(columns=col, inplace=True)
        if col in column_groups["q15_case_initiation"]:
            column_groups["q15_case_initiation"].remove(col)
    column_groups["q15_case_initiation"].append(pooled_column)


def build_feature_matrix(wide_csv: Path) -> FeatureMatrixArtifacts:
    df = _read_wide(wide_csv)
    column_groups = _collect_columns(df)

    ordered_columns: list[str] = []
    for group in ("base", "enums"):
        ordered_columns.extend(column_groups.get(group, []))
    for base in MULTI_BASES:
        ordered_columns.extend(column_groups.get(base, []))

    matrix = df.loc[:, ordered_columns].copy()
    _convert_indicator_columns(matrix, column_groups)
    _derive_notification_flags(matrix)
    _derive_power_features(matrix)
    _pool_case_initiation(matrix, column_groups)

    column_groups["derived_notification"] = [
        "art33_required_flag",
        "art33_submitted_flag",
        "art33_late_flag",
        "art33_timely_flag",
        "art34_required_flag",
        "subjects_notified_flag",
    ]
    column_groups["derived_power"] = [
        "power_option_count",
        "power_any_flag",
        "power_fine_flag",
        "power_warning_flag",
        "power_reprimand_flag",
        "power_none_flag",
        "power_combined_flag",
        "power_non_fine_flag",
    ]

    return FeatureMatrixArtifacts(matrix, column_groups)


def _save_outputs(artifacts: FeatureMatrixArtifacts, out_parquet: Path, out_metadata: Path) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    artifacts.dataframe.to_parquet(out_parquet, index=False)
    metadata = {
        "columns": list(artifacts.dataframe.columns),
        "column_groups": {key: list(value) for key, value in artifacts.column_groups.items()},
        "record_count": int(len(artifacts.dataframe)),
    }
    out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enriched feature matrix for breach notification analyses")
    parser.add_argument("--wide-csv", type=Path, default=Path("outputs/cleaned_wide.csv"), help="Path to cleaned wide CSV")
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("outputs/analysis/feature_matrix.parquet"),
        help="Destination parquet path",
    )
    parser.add_argument(
        "--out-metadata",
        type=Path,
        default=Path("outputs/analysis/feature_matrix_metadata.json"),
        help="Destination metadata JSON path",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = build_feature_matrix(args.wide_csv)
    _save_outputs(artifacts, args.out_parquet, args.out_metadata)


if __name__ == "__main__":
    main()
