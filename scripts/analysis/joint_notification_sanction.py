from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .build_feature_matrix import META_SUFFIXES

FEATURE_SETS: tuple[str, ...] = (
    "q10_org_class",
    "q15_case_initiation",
    "q21_breach_types",
    "q28_mitigations",
    "q46_vuln",
    "q47_remedial",
)

BASE_NUMERIC: tuple[str, ...] = (
    "n_principles_discussed",
    "n_principles_violated",
    "n_corrective_measures",
    "breach_case",
    "dpa_severity_shrinkage",
    "dpa_case_count",
)

CATEGORICAL: tuple[str, ...] = (
    "country_group",
    "isic_section",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint notification timeliness and sanction estimator")
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=Path("outputs/analysis/feature_matrix.parquet"),
        help="Path to feature matrix parquet",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("outputs/analysis/feature_matrix_metadata.json"),
        help="Path to feature matrix metadata",
    )
    parser.add_argument(
        "--hierarchy-json",
        type=Path,
        default=Path("outputs/analysis/hierarchical_severity/group_diagnostics.json"),
        help="Optional diagnostics JSON to augment with DPA shrinkage",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/joint_notification"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--latent-scores",
        type=Path,
        default=Path("outputs/analysis/interaction/latent_scores.parquet"),
        help="Optional latent component scores parquet to merge",
    )
    parser.add_argument(
        "--propensity-c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength for the propensity logistic model",
    )
    parser.add_argument(
        "--trim-lower",
        type=float,
        default=0.05,
        help="Lower propensity bound for trimming",
    )
    parser.add_argument(
        "--trim-upper",
        type=float,
        default=0.95,
        help="Upper propensity bound for trimming",
    )
    return parser.parse_args(argv)


def _load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_metadata(path: Path) -> dict[str, Sequence[str]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return metadata.get("column_groups", {})


def _ensure_hierarchy_features(df: pd.DataFrame, hierarchy_json: Path) -> pd.DataFrame:
    if hierarchy_json.exists():
        hierarchy = json.loads(hierarchy_json.read_text(encoding="utf-8"))
        mapping = {
            (item["dpa_name_canonical"], item.get("country_group")): item
            for item in hierarchy
        }
        shrink = []
        case_count = []
        for _, row in df.iterrows():
            key = (row.get("dpa_name_canonical"), row.get("country_group"))
            info = mapping.get(key)
            if info:
                shrink.append(info.get("expected_rank", np.nan))
                case_count.append(info.get("cases", np.nan))
            else:
                shrink.append(np.nan)
                case_count.append(np.nan)
        df = df.copy()
        df["dpa_severity_shrinkage"] = pd.Series(shrink).fillna(df.get("dpa_severity_shrinkage", df["n_principles_violated"].mean()))
        df["dpa_case_count"] = pd.Series(case_count).fillna(df.get("dpa_case_count", 1))
    else:
        if "dpa_severity_shrinkage" not in df.columns:
            df["dpa_severity_shrinkage"] = df["n_principles_violated"].mean()
        if "dpa_case_count" not in df.columns:
            df["dpa_case_count"] = 1
    return df


def _build_feature_frame(
    df: pd.DataFrame,
    metadata: dict[str, Sequence[str]],
    extra_columns: Sequence[str],
) -> pd.DataFrame:
    frame = df[list(BASE_NUMERIC)].copy()
    power_columns = [
        col
        for col in metadata.get("q53_powers", [])
        if not any(col.endswith(suffix) for suffix in META_SUFFIXES) and col in df.columns
    ]
    for key in FEATURE_SETS:
        columns = [
            col
            for col in metadata.get(key, [])
            if not any(col.endswith(suffix) for suffix in META_SUFFIXES) and col in df.columns
        ]
        if not columns:
            continue
        values = df[columns].fillna(0).astype(float)
        count_col = values.sum(axis=1)
        any_col = (count_col > 0).astype(float)
        frame[f"{key}_count"] = count_col
        frame[f"{key}_any"] = any_col
    for col in extra_columns:
        if col in df.columns:
            frame[col] = df[col]
    for cat in CATEGORICAL:
        frame[cat] = df[cat]
    if power_columns:
        frame = pd.concat([frame, df[power_columns].fillna(0).astype(float)], axis=1)
    return frame


def _prep_design_matrix(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.select_dtypes(include=["number"]).fillna(0).astype(float)
    categorical = pd.get_dummies(features[list(CATEGORICAL)].fillna("UNKNOWN"), drop_first=True, dtype=float)
    binary_sets = features.drop(columns=list(BASE_NUMERIC) + list(CATEGORICAL), errors="ignore").fillna(0).astype(float)
    design = pd.concat([numeric, categorical, binary_sets], axis=1)
    design = design.loc[:, ~design.columns.duplicated()]
    design = design.astype(float)
    return design


def _fit_propensity(y: pd.Series, X: pd.DataFrame, c: float) -> tuple[np.ndarray, pd.DataFrame]:
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        penalty="l2",
        C=c,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X_scaled, y)
    propensity = model.predict_proba(X_scaled)[:, 1]

    coef = []
    order = np.argsort(np.abs(model.coef_[0]))[::-1]
    for idx in order[:40]:
        coef.append(
            {
                "feature": X.columns[idx],
                "coefficient": float(model.coef_[0][idx]),
            }
        )
    return propensity, pd.DataFrame(coef)


def _aipw_estimate(y: pd.Series, treatment: pd.Series, propensity: np.ndarray, m1: np.ndarray, m0: np.ndarray) -> float:
    t = treatment.to_numpy()
    y_arr = y.to_numpy()
    clip = np.clip(propensity, 1e-3, 1 - 1e-3)
    term1 = m1 + (t / clip) * (y_arr - m1)
    term0 = m0 + ((1 - t) / (1 - clip)) * (y_arr - m0)
    return float(np.mean(term1 - term0))


def _propensity_diagnostics(propensity: np.ndarray) -> dict[str, object]:
    bins = np.linspace(0, 1, 11)
    counts, edges = np.histogram(propensity, bins=bins)
    quantiles = np.quantile(propensity, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "quantiles": {
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df = _load_matrix(args.feature_matrix)
    metadata = _load_metadata(args.metadata_json)
    df = _ensure_hierarchy_features(df, args.hierarchy_json)
    latent_columns: list[str] = []
    if args.latent_scores.exists():
        latent = pd.read_parquet(args.latent_scores)
        df = df.join(latent, how="left")
        latent_columns = list(latent.columns)

    subset = df[(df["art33_required_flag"] == 1) & df["art33_timely_flag"].notna()]
    subset = subset.copy()
    subset = subset[subset["fine_log1p"].notna()]

    features = _build_feature_frame(subset, metadata, latent_columns)
    design = _prep_design_matrix(subset, features)

    y_treat = subset["art33_timely_flag"].astype(float)
    y_outcome = subset["fine_log1p"].astype(float)

    propensity, prop_coefs = _fit_propensity(y_treat, design, args.propensity_c)
    mask = (propensity >= args.trim_lower) & (propensity <= args.trim_upper)

    trimmed_design = design.loc[mask]
    trimmed_treat = y_treat.loc[mask]
    trimmed_outcome = y_outcome.loc[mask]
    trimmed_propensity = propensity[mask]

    design_with_treatment = trimmed_design.assign(art33_timely_flag=trimmed_treat)
    design_with_treatment = sm.add_constant(design_with_treatment, has_constant="add")
    linear_model = sm.OLS(trimmed_outcome, design_with_treatment).fit()
    treated_design = design_with_treatment.copy()
    treated_design["art33_timely_flag"] = 1.0
    control_design = design_with_treatment.copy()
    control_design["art33_timely_flag"] = 0.0
    m1 = linear_model.predict(treated_design)
    m0 = linear_model.predict(control_design)

    aipw = _aipw_estimate(trimmed_outcome, trimmed_treat, trimmed_propensity, m1, m0)

    naive = float(trimmed_outcome[trimmed_treat == 1].mean() - trimmed_outcome[trimmed_treat == 0].mean())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "joint_model_metrics.json"
    propensity_path = args.out_dir / "propensity_diagnostics.json"
    prop_coef_path = args.out_dir / "propensity_top_coefficients.csv"

    metrics = {
        "initial_sample_size": int(len(subset)),
        "trimmed_sample_size": int(mask.sum()),
        "treated_cases": int(trimmed_treat.sum()),
        "control_cases": int((1 - trimmed_treat).sum()),
        "naive_average_treatment_effect": naive,
        "aipw_average_treatment_effect": aipw,
        "propensity_mean": float(trimmed_propensity.mean()),
        "propensity_std": float(trimmed_propensity.std()),
        "trim_lower": args.trim_lower,
        "trim_upper": args.trim_upper,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    propensity_stats = _propensity_diagnostics(trimmed_propensity)
    propensity_path.write_text(json.dumps(propensity_stats, indent=2), encoding="utf-8")
    prop_coefs.to_csv(prop_coef_path, index=False)


if __name__ == "__main__":
    main()
