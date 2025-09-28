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

from .joint_notification_sanction import (
    _load_matrix,
    _load_metadata,
    _ensure_hierarchy_features,
    _build_feature_frame,
)
from .build_feature_matrix import META_SUFFIXES


def parse_trim_grid(spec: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for segment in spec.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid trim specification: {segment}")
        lower, upper = float(parts[0]), float(parts[1])
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(f"Trim bounds must satisfy 0 <= lower < upper <= 1: {segment}")
        pairs.append((lower, upper))
    if not pairs:
        raise ValueError("At least one trim pair must be provided")
    return pairs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap AIPW estimates for Art. 33 notification timeliness")
    parser.add_argument("--feature-matrix", type=Path, default=Path("outputs/analysis/feature_matrix.parquet"))
    parser.add_argument("--metadata-json", type=Path, default=Path("outputs/analysis/feature_matrix_metadata.json"))
    parser.add_argument("--latent-scores", type=Path, default=Path("outputs/analysis/interaction/latent_scores.parquet"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/analysis/joint_notification/bootstrap_results.json"))
    parser.add_argument("--propensity-c", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--trim-grid", type=str, default="0.1,0.9")
    return parser.parse_args(argv)


def _prep_design(df: pd.DataFrame, metadata: dict[str, Sequence[str]], extra_columns: Sequence[str]) -> pd.DataFrame:
    frame = _build_feature_frame(df, metadata, extra_columns)
    numeric = frame.select_dtypes(include=["number"]).fillna(0).astype(float)
    categorical_cols = [col for col in frame.columns if frame[col].dtype == "object"]
    categorical = pd.get_dummies(frame[categorical_cols].fillna("UNKNOWN"), drop_first=True, dtype=float) if categorical_cols else pd.DataFrame(index=frame.index)
    binary = frame.drop(columns=list(numeric.columns) + categorical_cols, errors="ignore").fillna(0).astype(float)
    design = pd.concat([numeric, categorical, binary], axis=1)
    design = design.loc[:, ~design.columns.duplicated()]
    return design.astype(float)


def _fit_propensity(design: pd.DataFrame, treatment: pd.Series, c: float) -> np.ndarray:
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(design)
    model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=1000)
    model.fit(X_scaled, treatment)
    return model.predict_proba(X_scaled)[:, 1]


def _fit_linear(design: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    augmented = design.copy()
    augmented["art33_timely_flag"] = treatment
    augmented = sm.add_constant(augmented, has_constant="add")
    return sm.OLS(outcome, augmented).fit()


def _predict_linear(model: sm.regression.linear_model.RegressionResultsWrapper, design: pd.DataFrame, treatment_value: float) -> np.ndarray:
    augmented = design.copy()
    augmented["art33_timely_flag"] = treatment_value
    augmented = sm.add_constant(augmented, has_constant="add")
    return model.predict(augmented)


def _aipw(y: pd.Series, treatment: pd.Series, propensity: np.ndarray, m1: np.ndarray, m0: np.ndarray) -> float:
    clip = np.clip(propensity, 1e-3, 1 - 1e-3)
    treated = treatment.to_numpy()
    outcome = y.to_numpy()
    term1 = m1 + (treated / clip) * (outcome - m1)
    term0 = m0 + ((1 - treated) / (1 - clip)) * (outcome - m0)
    return float(np.mean(term1 - term0))


def _prepare_latent_columns(df: pd.DataFrame, latent_scores: Path) -> list[str]:
    if latent_scores.exists():
        latent = pd.read_parquet(latent_scores)
        df.loc[:, latent.columns] = latent
        return list(latent.columns)
    return []


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    trim_pairs = parse_trim_grid(args.trim_grid)

    df = _load_matrix(args.feature_matrix)
    metadata = _load_metadata(args.metadata_json)
    df = _ensure_hierarchy_features(df, Path("outputs/analysis/hierarchical_severity/group_diagnostics.json"))
    df = df[(df["art33_required_flag"] == 1) & df["art33_timely_flag"].notna() & df["fine_log1p"].notna()]

    latent_columns = _prepare_latent_columns(df, args.latent_scores)
    design = _prep_design(df, metadata, latent_columns)

    treatment = df["art33_timely_flag"].astype(float)
    outcome = df["fine_log1p"].astype(float)

    results: dict[str, dict[str, float]] = {}

    for lower, upper in trim_pairs:
        propensity = _fit_propensity(design, treatment, args.propensity_c)
        mask = (propensity >= lower) & (propensity <= upper)

        trimmed_design = design.loc[mask]
        trimmed_treat = treatment.loc[mask]
        trimmed_outcome = outcome.loc[mask]
        trimmed_propensity = propensity[mask]

        if trimmed_design.empty:
            results[f"{lower}-{upper}"] = {
                "trimmed_sample_size": 0,
                "treated_cases": 0,
                "control_cases": 0,
                "naive_effect": float("nan"),
                "aipw_effect": float("nan"),
                "bootstrap_mean": float("nan"),
                "bootstrap_std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
            continue

        base_linear = _fit_linear(trimmed_design, trimmed_treat, trimmed_outcome)
        m1 = _predict_linear(base_linear, trimmed_design, 1.0)
        m0 = _predict_linear(base_linear, trimmed_design, 0.0)
        aipw = _aipw(trimmed_outcome, trimmed_treat, trimmed_propensity, m1, m0)
        naive = float(trimmed_outcome[trimmed_treat == 1].mean() - trimmed_outcome[trimmed_treat == 0].mean())

        bootstrap_estimates = []
        trimmed_index = trimmed_design.index.to_list()

        for _ in range(args.bootstrap):
            sample_indices = np.random.choice(trimmed_index, size=len(trimmed_index), replace=True)
            sample_design = trimmed_design.loc[sample_indices].reset_index(drop=True)
            sample_treat = trimmed_treat.loc[sample_indices].reset_index(drop=True)
            sample_outcome = trimmed_outcome.loc[sample_indices].reset_index(drop=True)

            sample_propensity = _fit_propensity(sample_design, sample_treat, args.propensity_c)
            sample_linear = _fit_linear(sample_design, sample_treat, sample_outcome)
            m1_b = _predict_linear(sample_linear, sample_design, 1.0)
            m0_b = _predict_linear(sample_linear, sample_design, 0.0)
            estimate = _aipw(sample_outcome, sample_treat, sample_propensity, m1_b, m0_b)
            bootstrap_estimates.append(estimate)

        est_array = np.array(bootstrap_estimates)
        results[f"{lower}-{upper}"] = {
            "trimmed_sample_size": int(mask.sum()),
            "treated_cases": int(trimmed_treat.sum()),
            "control_cases": int((1 - trimmed_treat).sum()),
            "naive_effect": naive,
            "aipw_effect": aipw,
            "bootstrap_mean": float(est_array.mean()),
            "bootstrap_std": float(est_array.std(ddof=1)),
            "ci_lower": float(np.quantile(est_array, 0.025)),
            "ci_upper": float(np.quantile(est_array, 0.975)),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
