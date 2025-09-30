"""Rigorous analysis of Art. 33 breach notification behaviour and enforcement outcomes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path(__file__).resolve().parents[1] / "outputs" / "cleaned_wide.csv"
RESULTS_JSON = Path(__file__).resolve().parent / "breach_notification_case_results.json"
REGRESSIONS_CSV = Path(__file__).resolve().parent / "breach_notification_case_regressions.csv"


@dataclass
class ModelResult:
    name: str
    outcome: str
    n: int
    coefficients: pd.DataFrame

    def to_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for term, row in self.coefficients.iterrows():
            records.append(
                {
                    "model": self.name,
                    "outcome": self.outcome,
                    "term": term,
                    "estimate": row["coef"],
                    "std_err": row["std_err"],
                    "t_value": row.get("t_value"),
                    "p_value": row.get("p_value"),
                    "n_obs": self.n,
                }
            )
        return records


def _map_flag(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    mapped = series.map(mapping)
    return mapped.astype(float)


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    breach = df[df["breach_case"] == 1].copy()

    breach["art33_required_flag"] = _map_flag(
        breach["art33_notification_required"], {"YES_REQUIRED": 1, "NO_NOT_REQUIRED": 0}
    )
    breach["art33_notified_flag"] = _map_flag(
        breach["art33_notification_submitted"], {"YES_SUBMITTED": 1, "NO_NOT_SUBMITTED": 0}
    )
    breach["art33_timely_flag"] = _map_flag(
        breach["art33_notification_timeliness"], {"YES_WITHIN_72H": 1, "NO_LATE": 0}
    )

    delay_mapping = {
        "1_TO_4_WEEKS": 28,
        "1_TO_6_MONTHS": 120,
        "OVER_6_MONTHS": 210,
    }
    breach["art33_delay_days"] = breach["art33_notification_delay_band"].map(delay_mapping)

    breach["case_initiated_by_notification"] = breach["q15_case_initiation_BREACH_NOTIFICATION"].fillna(0)
    breach["enforcement_severity_index"] = breach["fine_positive"] + breach["severity_measures_present"]

    breach["decision_year_cat"] = breach["decision_year"].fillna(-1).astype(int).astype(str)
    breach.loc[breach["decision_year"].isna(), "decision_year_cat"] = "Missing"

    return breach


def build_feature_matrix(df: pd.DataFrame, include_initiation: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    base_cols = [
        "q21_breach_types_TECHNICAL_FAILURE",
        "q21_breach_types_ORGANIZATIONAL_FAILURE",
        "q21_breach_types_CYBER_ATTACK",
        "q21_breach_types_HUMAN_ERROR",
        "q21_breach_types_SYSTEM_MALFUNCTION",
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
        "q25_sensitive_data_ARTICLE_10_CRIMINAL",
        "q25_sensitive_data_NEITHER",
        "n_principles_discussed",
        "n_principles_violated",
    ]

    # Some breach cases mark sensitive-data question as not applicable; fill NaN with 0 to keep indicators consistent.
    features = df[base_cols].fillna(0).copy()

    cat_cols = ["country_code", "isic_section_desc", "decision_year_cat"]
    categorical = pd.get_dummies(df[cat_cols], drop_first=True, dummy_na=True)
    features = pd.concat([features.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)

    if include_initiation:
        features = pd.concat(
            [features, df[["case_initiated_by_notification"]].reset_index(drop=True)], axis=1
        )

    return features, list(features.columns)


def fit_ols(name: str, df: pd.DataFrame, y_col: str, X: pd.DataFrame) -> ModelResult:
    y = df[y_col].reset_index(drop=True)
    design = pd.concat(
        [df[["art33_notified_flag"]].reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )
    design = design.astype(float)
    X_design = sm.add_constant(design)
    model = sm.OLS(y, X_design, missing="drop")
    fitted = model.fit(cov_type="HC3")
    summary_df = pd.DataFrame(
        {
            "coef": fitted.params,
            "std_err": fitted.bse,
            "t_value": fitted.tvalues,
            "p_value": fitted.pvalues,
        }
    )
    return ModelResult(name=name, outcome=y_col, n=int(fitted.nobs), coefficients=summary_df)


def fit_glm_binomial(name: str, df: pd.DataFrame, y_col: str, X: pd.DataFrame) -> ModelResult:
    y = df[y_col].reset_index(drop=True)
    design = pd.concat(
        [df[["art33_notified_flag"]].reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )
    design = design.astype(float)
    X_design = sm.add_constant(design)
    model = sm.GLM(y, X_design, family=sm.families.Binomial())
    fitted = model.fit(cov_type="HC3")
    summary_df = pd.DataFrame(
        {
            "coef": fitted.params,
            "std_err": fitted.bse,
            "t_value": fitted.tvalues,
            "p_value": fitted.pvalues,
        }
    )
    return ModelResult(name=name, outcome=y_col, n=int(fitted.nobs), coefficients=summary_df)


def fit_glm_poisson(name: str, df: pd.DataFrame, y_col: str, X: pd.DataFrame) -> ModelResult:
    y = df[y_col].reset_index(drop=True)
    design = pd.concat(
        [df[["art33_notified_flag"]].reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )
    design = design.astype(float)
    X_design = sm.add_constant(design)
    model = sm.GLM(y, X_design, family=sm.families.Poisson())
    fitted = model.fit(cov_type="HC3")
    summary_df = pd.DataFrame(
        {
            "coef": fitted.params,
            "std_err": fitted.bse,
            "t_value": fitted.tvalues,
            "p_value": fitted.pvalues,
        }
    )
    return ModelResult(name=name, outcome=y_col, n=int(fitted.nobs), coefficients=summary_df)


def inverse_probability_weighting(
    df: pd.DataFrame,
    outcome_col: str,
    features: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int | None = 42,
) -> Dict[str, float]:
    treated = df["art33_notified_flag"].values
    outcome = df[outcome_col].values

    if treated.sum() == 0 or treated.sum() == len(treated):
        raise ValueError("Treatment indicator lacks variation")

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(features.values, treated)
    propensity = clf.predict_proba(features.values)[:, 1]
    propensity = np.clip(propensity, 0.05, 0.95)

    def _weighted_difference(idx: np.ndarray) -> float:
        t = treated[idx]
        y = outcome[idx]
        p = propensity[idx]
        w_t = t / p
        w_c = (1 - t) / (1 - p)
        mu_t = np.sum(w_t * y) / np.sum(w_t)
        mu_c = np.sum(w_c * y) / np.sum(w_c)
        return mu_t - mu_c

    ate = _weighted_difference(np.arange(len(df)))

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(df), len(df))
        boot.append(_weighted_difference(sample_idx))

    boot = np.array(boot)
    return {
        "ate": float(ate),
        "bootstrap_mean": float(boot.mean()),
        "ci_lower": float(np.percentile(boot, 2.5)),
        "ci_upper": float(np.percentile(boot, 97.5)),
        "n_obs": int(len(df)),
    }


def assemble_descriptives(breach: pd.DataFrame) -> Dict[str, object]:
    counts = {
        "breach_cases": int(len(breach)),
        "art33_known_required": int(breach["art33_required_flag"].notna().sum()),
        "art33_required_yes": int((breach["art33_required_flag"] == 1).sum()),
        "art33_required_no": int((breach["art33_required_flag"] == 0).sum()),
    }

    required = breach[breach["art33_required_flag"] == 1].copy()
    submitted = required[required["art33_notified_flag"].notna()].copy()

    group_stats = {}
    for flag_value, group in submitted.groupby("art33_notified_flag"):
        key = "notified" if flag_value == 1 else "not_notified"
        group_stats[key] = {
            "cases": int(len(group)),
            "fine_positive_rate": float(group["fine_positive"].mean()),
            "median_fine_eur": float(group["fine_eur"].median()),
            "mean_fine_log1p": float(group["fine_log1p"].mean()),
            "mean_corrective_count": float(group["n_corrective_measures"].mean()),
            "severity_index_mean": float(group["enforcement_severity_index"].mean()),
            "initiation_by_notification_rate": float(group["case_initiated_by_notification"].mean()),
        }

    timing = submitted[submitted["art33_timely_flag"].notna()].copy()
    timing_stats = {}
    for flag_value, group in timing.groupby("art33_timely_flag"):
        key = "timely" if flag_value == 1 else "late"
        timing_stats[key] = {
            "cases": int(len(group)),
            "fine_positive_rate": float(group["fine_positive"].mean()),
            "median_fine_eur": float(group["fine_eur"].median()),
            "mean_fine_log1p": float(group["fine_log1p"].mean()),
            "mean_corrective_count": float(group["n_corrective_measures"].mean()),
            "severity_index_mean": float(group["enforcement_severity_index"].mean()),
        }

    return {
        "counts": counts,
        "required_stats": group_stats,
        "timing_stats": timing_stats,
    }


def run_analysis() -> None:
    breach = load_and_prepare()

    descriptives = assemble_descriptives(breach)

    required = breach[breach["art33_required_flag"] == 1].copy()
    model_df = required[required["art33_notified_flag"].notna()].copy()

    features, feature_names = build_feature_matrix(model_df, include_initiation=False)

    regression_results: List[ModelResult] = []
    regression_results.append(fit_ols("OLS_FineLog", model_df, "fine_log1p", features))
    regression_results.append(fit_ols("OLS_FinePositive", model_df, "fine_positive", features))
    regression_results.append(fit_ols("OLS_CorrectiveCount", model_df, "n_corrective_measures", features))

    # Propensity-weighted estimate for fine_log1p
    ipw_info = inverse_probability_weighting(model_df, "fine_log1p", features)

    # Save artefacts
    results_payload = {
        "descriptives": descriptives,
        "ipw_fine_log1p": ipw_info,
        "feature_columns": feature_names,
    }
    RESULTS_JSON.write_text(json.dumps(results_payload, indent=2))

    rows: List[Dict[str, object]] = []
    for result in regression_results:
        rows.extend(result.to_records())
    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(REGRESSIONS_CSV, index=False)


if __name__ == "__main__":
    run_analysis()
