from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .build_feature_matrix import META_SUFFIXES, POWER_TOKEN_COLUMNS

SEVERITY_ORDER = {
    "NONE": 0,
    "WARNING_REPRIMAND": 1,
    "REMEDIAL_ONLY": 2,
    "FINE_ONLY": 3,
    "FINE_PLUS": 4,
}

MULTI_FEATURE_KEYS: tuple[str, ...] = (
    "q10_org_class",
    "q21_breach_types",
    "q28_mitigations",
    "q46_vuln",
    "q47_remedial",
)

BASIC_FEATURES: tuple[str, ...] = (
    "art33_required_flag",
    "art33_submitted_flag",
    "art33_timely_flag",
    "art34_required_flag",
    "subjects_notified_flag",
    "breach_case",
    "n_principles_discussed",
    "n_principles_violated",
    "n_corrective_measures",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "country_group",
    "isic_section",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an ordinal severity model with DPA-level shrinkage features"
    )
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=Path("outputs/analysis/feature_matrix.parquet"),
        help="Path to the feature matrix parquet",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("outputs/analysis/feature_matrix_metadata.json"),
        help="Path to accompanying metadata JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/hierarchical_severity"),
        help="Directory for model artefacts",
    )
    parser.add_argument(
        "--max-dpa-dummies",
        type=int,
        default=20,
        help="Number of most frequent DPAs to encode explicitly (others pooled)",
    )
    parser.add_argument(
        "--latent-scores",
        type=Path,
        default=Path("outputs/analysis/interaction/latent_scores.parquet"),
        help="Optional latent component scores parquet to merge as covariates",
    )
    return parser.parse_args(argv)


def _load_inputs(feature_matrix: Path, metadata_json: Path) -> tuple[pd.DataFrame, dict[str, Sequence[str]]]:
    df = pd.read_parquet(feature_matrix)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    column_groups: dict[str, Sequence[str]] = metadata.get("column_groups", {})
    return df, column_groups


def _severity_label(row: pd.Series) -> str:
    any_flag = row.get("power_any_flag", 0)
    none_flag = row.get("power_none_flag", 0)
    fine_flag = row.get("power_fine_flag", 0)
    combined_flag = row.get("power_combined_flag", 0)
    warning_flag = row.get("power_warning_flag", 0)
    reprimand_flag = row.get("power_reprimand_flag", 0)
    non_fine_flag = row.get("power_non_fine_flag", 0)

    if pd.isna(any_flag):
        any_flag = 0
    if pd.isna(none_flag):
        none_flag = 0
    if pd.isna(fine_flag):
        fine_flag = 0
    if pd.isna(combined_flag):
        combined_flag = 0
    if pd.isna(warning_flag):
        warning_flag = 0
    if pd.isna(reprimand_flag):
        reprimand_flag = 0
    if pd.isna(non_fine_flag):
        non_fine_flag = 0

    if not any_flag and none_flag == 1:
        return "NONE"
    if fine_flag == 1:
        if combined_flag == 1:
            return "FINE_PLUS"
        return "FINE_ONLY"
    reprimand_warning = max(warning_flag, reprimand_flag) == 1
    if reprimand_warning and not non_fine_flag:
        return "WARNING_REPRIMAND"
    if non_fine_flag:
        return "REMEDIAL_ONLY"
    return "NONE"


def _build_multi_features(
    df: pd.DataFrame,
    column_groups: dict[str, Sequence[str]],
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for key in MULTI_FEATURE_KEYS:
        columns = [
            col
            for col in column_groups.get(key, [])
            if not any(col.endswith(suffix) for suffix in META_SUFFIXES)
            and col not in POWER_TOKEN_COLUMNS
            and col in df.columns
        ]
        if not columns:
            continue
        values = df[columns].fillna(0).astype(float)
        count_col = values.sum(axis=1)
        any_col = (count_col > 0).astype(float)
        features[f"{key}_count"] = count_col
        features[f"{key}_any"] = any_col
    if not features:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(features, index=df.index)


def _dpa_shrinkage_features(df: pd.DataFrame, severity_rank: pd.Series) -> pd.DataFrame:
    frame = df[["dpa_name_canonical"]].copy()
    frame["severity_rank"] = severity_rank
    grouped = frame.groupby("dpa_name_canonical")
    totals = grouped["severity_rank"].agg(["mean", "count"])
    global_mean = severity_rank.mean()
    shrinkage = {}
    for dpa, row in totals.iterrows():
        count = row["count"]
        mean_value = row["mean"]
        alpha = max(count - 1, 1)
        shrinkage[dpa] = (alpha * mean_value + global_mean) / (alpha + 1)
    df = df.copy()
    df["dpa_severity_shrinkage"] = df["dpa_name_canonical"].map(shrinkage).fillna(global_mean)
    df["dpa_case_count"] = df["dpa_name_canonical"].map(totals["count"]).fillna(1)
    return df[["dpa_severity_shrinkage", "dpa_case_count"]]


def _encode_categorical(df: pd.DataFrame, max_dpa_dummies: int) -> pd.DataFrame:
    top_dpas = df["dpa_name_canonical"].value_counts().head(max_dpa_dummies).index
    df["dpa_group"] = np.where(df["dpa_name_canonical"].isin(top_dpas), df["dpa_name_canonical"], "OTHER")
    categorical = pd.get_dummies(df["dpa_group"], prefix="dpa", drop_first=True, dtype=float)
    for cat in CATEGORICAL_FEATURES:
        categorical = pd.concat(
            [categorical, pd.get_dummies(df[cat].fillna("UNKNOWN"), prefix=cat, drop_first=True, dtype=float)],
            axis=1,
        )
    return categorical


def _assemble_design_matrix(
    df: pd.DataFrame,
    multi_features: pd.DataFrame,
    latent_columns: list[str],
    max_dpa_dummies: int,
) -> pd.DataFrame:
    base = df[list(BASIC_FEATURES)].copy()
    base = base.fillna(0).astype(float)

    multi = multi_features.copy()
    if not multi.empty:
        multi = multi.fillna(0).astype(float)
    else:
        multi = pd.DataFrame(index=df.index)

    cat = _encode_categorical(df, max_dpa_dummies)
    shrink = _dpa_shrinkage_features(df, df["severity_rank"])

    latent = pd.DataFrame(index=df.index)
    if latent_columns:
        latent = df[latent_columns].fillna(0).astype(float)

    design = pd.concat([base, multi, cat, shrink, latent], axis=1)
    # Ensure no duplicated columns
    design = design.loc[:, ~design.columns.duplicated()]
    variances = design.var(axis=0)
    keep = variances[variances > 1e-6].index
    design = design.loc[:, keep]
    return design


def _fit_ordered_model(severity_rank: pd.Series, design: pd.DataFrame) -> OrderedModel:
    model = OrderedModel(severity_rank, design, distr="logit")
    return model


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df, column_groups = _load_inputs(args.feature_matrix, args.metadata_json)
    latent_columns: list[str] = []
    if args.latent_scores.exists():
        latent = pd.read_parquet(args.latent_scores)
        df = df.join(latent, how="left")
        latent_columns = list(latent.columns)

    df = df.copy()

    df["severity_label"] = df.apply(_severity_label, axis=1)
    df["severity_rank"] = df["severity_label"].map(SEVERITY_ORDER)
    df = df.dropna(subset=["severity_rank"]).reset_index(drop=True)
    df["severity_rank"] = df["severity_rank"].astype(int)

    multi_features = _build_multi_features(df, column_groups)
    design = _assemble_design_matrix(df, multi_features, latent_columns, args.max_dpa_dummies)

    design = design.replace({pd.NA: 0}).fillna(0).astype(float)

    model = _fit_ordered_model(df["severity_rank"], design)
    result = model.fit(method="bfgs", disp=False, maxiter=200)

    preds = result.model.predict(result.params, exog=design, which="prob")
    observed_levels = np.sort(df["severity_rank"].unique())
    severity_levels = np.array(observed_levels[: preds.shape[1]])
    expected = (preds * severity_levels).sum(axis=1)
    df["severity_expected"] = expected

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "ordinal_model_summary.txt"
    predictions_path = args.out_dir / "severity_predictions.csv"
    diagnostics_path = args.out_dir / "group_diagnostics.json"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Ordinal logit severity model\n")
        f.write(f"Sample size: {len(df)}\n")
        f.write(f"Features: {design.shape[1]}\n\n")
        f.write(result.summary().as_text())

    output = df[
        [
            "decision_id",
            "dpa_name_canonical",
            "country_group",
            "severity_label",
            "severity_rank",
            "severity_expected",
            "art33_timely_flag",
            "subjects_notified_flag",
        ]
    ].copy()
    output.to_csv(predictions_path, index=False)

    group_stats = (
        output.groupby(["dpa_name_canonical", "country_group"])
        .agg(
            cases=("decision_id", "count"),
            average_rank=("severity_rank", "mean"),
            expected_rank=("severity_expected", "mean"),
        )
        .reset_index()
        .sort_values(by="average_rank", ascending=False)
    )
    diagnostics_path.write_text(group_stats.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
