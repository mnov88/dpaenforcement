from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .build_feature_matrix import (
    META_SUFFIXES,
    POWER_TOKEN_COLUMNS,
)

NOTIFICATION_FLAGS: tuple[str, ...] = (
    "art33_required_flag",
    "art33_submitted_flag",
    "art33_timely_flag",
    "art33_late_flag",
    "art34_required_flag",
    "subjects_notified_flag",
)

POWER_FLAGS: tuple[str, ...] = (
    "power_any_flag",
    "power_fine_flag",
    "power_warning_flag",
    "power_reprimand_flag",
    "power_none_flag",
    "power_combined_flag",
    "power_non_fine_flag",
)

COVERAGE_PREFIXES: dict[str, str] = {
    "org_class": "q10_org_class_",
    "breach_types": "q21_breach_types_",
    "mitigations": "q28_mitigations_",
    "vulnerabilities": "q46_vuln_",
}


def _load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("feature matrix is empty; run build_feature_matrix first")
    return df


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    result: dict[str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            label = "NA"
        else:
            label = str(int(key)) if isinstance(key, (int, bool)) else str(key)
        result[label] = int(value)
    return result


def _summarise_notification_flags(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {flag: _value_counts(df.get(flag, pd.Series(dtype="Int64"))) for flag in NOTIFICATION_FLAGS}


def _summarise_power_tokens(df: pd.DataFrame) -> dict[str, int]:
    summary: dict[str, int] = {}
    for column in POWER_TOKEN_COLUMNS:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            summary[column] = int(values.sum())
        else:
            summary[column] = 0
    return summary


def _summarise_power_flags(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {flag: _value_counts(df.get(flag, pd.Series(dtype="Int64"))) for flag in POWER_FLAGS}


def _power_combinations(df: pd.DataFrame, top_n: int = 20) -> list[dict[str, object]]:
    present_columns = [col for col in POWER_TOKEN_COLUMNS if col in df.columns]
    if not present_columns:
        return []
    indicators = df[present_columns].fillna(0)
    combos = []
    for _, row in indicators.iterrows():
        active = [col.split("q53_powers_")[1] for col, val in row.items() if val > 0]
        combos.append("|".join(active) if active else "NONE")
    combo_counts = pd.Series(combos).value_counts().head(top_n)
    output: list[dict[str, object]] = []
    for combo, count in combo_counts.items():
        output.append({"combo": combo, "count": int(count)})
    return output


def _power_notification_crosstabs(df: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    crosstabs: dict[str, dict[str, dict[str, int]]] = {}
    for power in POWER_FLAGS:
        if power not in df.columns:
            continue
        crosstabs[power] = {}
        power_series = df[power]
        for notif in NOTIFICATION_FLAGS:
            if notif not in df.columns:
                continue
            notif_series = df[notif]
            table = pd.crosstab(
                power_series.fillna(-1),
                notif_series.fillna(-1),
                dropna=False,
            )
            crosstabs[power][notif] = {
                str(int(idx)) if idx != -1 else "NA": {str(int(col)) if col != -1 else "NA": int(val)
                                                        for col, val in row.items()}
                for idx, row in table.iterrows()
            }
    return crosstabs


def _coverage_summary(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for label, prefix in COVERAGE_PREFIXES.items():
        columns = [col for col in df.columns if col.startswith(prefix) and not col.endswith(META_SUFFIXES)]
        if not columns:
            summary[label] = {"columns": [], "coverage_count": 0, "coverage_ratio": 0.0}
            continue
        matrix = df[columns].fillna(0)
        any_presence = (matrix.sum(axis=1) > 0)
        count = int(any_presence.sum())
        ratio = float(count / len(df)) if len(df) > 0 else 0.0
        summary[label] = {
            "columns": columns,
            "coverage_count": count,
            "coverage_ratio": ratio,
        }
    return summary


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostics on the breach notification feature matrix")
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=Path("outputs/analysis/feature_matrix.parquet"),
        help="Path to the parquet feature matrix",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/diagnostics"),
        help="Directory to write diagnostic artefacts",
    )
    parser.add_argument(
        "--top-combos",
        type=int,
        default=20,
        help="Number of most frequent power combinations to record",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df = _load_matrix(args.feature_matrix)

    notif = _summarise_notification_flags(df)
    power_tokens = _summarise_power_tokens(df)
    power_flags = _summarise_power_flags(df)
    combos = _power_combinations(df, top_n=args.top_combos)
    crosstabs = _power_notification_crosstabs(df)
    coverage = _coverage_summary(df)

    _write_json(notif, args.out_dir / "notification_flags.json")
    _write_json(power_tokens, args.out_dir / "power_token_counts.json")
    _write_json(power_flags, args.out_dir / "power_flag_counts.json")
    _write_json(combos, args.out_dir / "power_combinations.json")
    _write_json(crosstabs, args.out_dir / "power_notification_crosstabs.json")
    _write_json(coverage, args.out_dir / "coverage_summary.json")


if __name__ == "__main__":
    main()
