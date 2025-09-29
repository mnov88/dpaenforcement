"""Leniency and severity index construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

@dataclass
class LeniencyResult:
    frame: pd.DataFrame
    base_model: object
    dpa_model: object
    country_model: object


def _make_formula(outcome: str, features: Sequence[str]) -> str:
    rhs = " + ".join(features)
    return f"{outcome} ~ {rhs}" if rhs else f"{outcome} ~ 1"


def compute_leniency_index(
    data: pd.DataFrame,
    outcome: str = "fine_log1p",
    fact_features: Sequence[str] | None = None,
    random_slope_terms: Sequence[str] | None = None,
) -> LeniencyResult:
    fact_features = fact_features or [
        "breach_case",
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
        "q25_sensitive_data_ARTICLE_10_CRIMINAL",
        "q25_sensitive_data_NEITHER",
        "q46_vuln_CHILDREN",
        "organization_size_tier",
        "organization_type",
        "case_origin",
        "n_principles_violated",
        "n_corrective_measures",
        "days_since_gdpr",
    ]
    # Prepare design matrix: cast categoricals and coerce numeric features, replacing pd.NA with np.nan
    safe = data.copy()
    # Replace pandas NA with numpy nan to avoid patsy's NA ambiguity in numerical contexts
    try:
        safe = safe.replace({pd.NA: np.nan})
    except Exception:
        pass
    categorical_cols = {"breach_case", "organization_size_tier", "organization_type", "case_origin", "dpa_name_canonical", "country_code"}
    for col in categorical_cols:
        if col in safe.columns:
            safe[col] = safe[col].astype("category")
    # Ensure numeric predictors are numeric with np.nan (not pandas NA)
    for col in fact_features:
        if col in safe.columns and col not in categorical_cols:
            arr = pd.to_numeric(safe[col], errors="coerce")
            # Force numpy float dtype so missing values are np.nan (not pandas NA)
            safe[col] = arr.astype(float)
    if outcome in safe.columns:
        safe[outcome] = pd.to_numeric(safe[outcome], errors="coerce").astype(float)
    formula = _make_formula(outcome, fact_features)
    base_model = smf.ols(formula, data=safe).fit()
    residuals = base_model.resid
    # Align working frame strictly to the rows used in OLS to avoid shape mismatches downstream
    working = safe.loc[residuals.index].copy()
    working["leniency_residual"] = residuals

    random_slope_terms = list(random_slope_terms or [])
    re_formula = "1"
    if random_slope_terms:
        re_formula = "1 + " + " + ".join(random_slope_terms)

    # Drop rows with missing groups or residuals
    working = working.dropna(subset=["leniency_residual", "dpa_name_canonical", "country_code"])
    # Try MixedLM; fallback to group means if singular/failed
    dpa_model = None
    country_model = None
    try:
        dpa_model = smf.mixedlm(
            "leniency_residual ~ 1",
            working,
            groups=working["dpa_name_canonical"],
            re_formula=re_formula,
        ).fit(reml=True, method="lbfgs", maxiter=500)
    except Exception:
        try:
            dpa_model = smf.mixedlm(
                "leniency_residual ~ 1",
                working,
                groups=working["dpa_name_canonical"],
            ).fit(reml=True, method="lbfgs", maxiter=500)
        except Exception:
            dpa_model = None

    try:
        country_model = smf.mixedlm(
            "leniency_residual ~ 1",
            working,
            groups=working["country_code"],
        ).fit(reml=True, method="lbfgs", maxiter=500)
    except Exception:
        country_model = None

    # SD estimates: prefer model-based; else fallback to within-group SD of residuals
    if dpa_model is not None and getattr(dpa_model, "cov_re", None) is not None and dpa_model.cov_re.size:
        dpa_var = float(dpa_model.cov_re.iloc[0, 0])
        dpa_sd = float(np.sqrt(dpa_var)) if np.isfinite(dpa_var) else float("nan")
    else:
        dpa_sd = float(working.groupby("dpa_name_canonical")["leniency_residual"].std(ddof=1).mean())
    if country_model is not None and getattr(country_model, "cov_re", None) is not None and country_model.cov_re.size:
        country_var = float(country_model.cov_re.iloc[0, 0])
        country_sd = float(np.sqrt(country_var)) if np.isfinite(country_var) else float("nan")
    else:
        country_sd = float(working.groupby("country_code")["leniency_residual"].std(ddof=1).mean())

    records: list[dict[str, object]] = []
    if dpa_model is not None and hasattr(dpa_model, "random_effects"):
        for dpa, effect in dpa_model.random_effects.items():
            value = float(effect[0]) if hasattr(effect, "__iter__") else float(effect)
            records.append(
                {
                    "jurisdiction_level": "DPA",
                    "jurisdiction": dpa,
                    "effect": value,
                    "sd": dpa_sd,
                    "lower": value - 1.96 * dpa_sd if np.isfinite(dpa_sd) else np.nan,
                    "upper": value + 1.96 * dpa_sd if np.isfinite(dpa_sd) else np.nan,
                    "n_obs": int((working["dpa_name_canonical"] == dpa).sum()),
                }
            )
    else:
        # Fallback: group means as effects
        grp = working.groupby("dpa_name_canonical")["leniency_residual"]
        for dpa, val in grp.mean().items():
            n = int((working["dpa_name_canonical"] == dpa).sum())
            records.append(
                {
                    "jurisdiction_level": "DPA",
                    "jurisdiction": dpa,
                    "effect": float(val),
                    "sd": dpa_sd,
                    "lower": float(val) - 1.96 * dpa_sd if np.isfinite(dpa_sd) else np.nan,
                    "upper": float(val) + 1.96 * dpa_sd if np.isfinite(dpa_sd) else np.nan,
                    "n_obs": n,
                }
            )

    if country_model is not None and hasattr(country_model, "random_effects"):
        for country, effect in country_model.random_effects.items():
            value = float(effect[0]) if hasattr(effect, "__iter__") else float(effect)
            records.append(
                {
                    "jurisdiction_level": "Country",
                    "jurisdiction": country,
                    "effect": value,
                    "sd": country_sd,
                    "lower": value - 1.96 * country_sd if np.isfinite(country_sd) else np.nan,
                    "upper": value + 1.96 * country_sd if np.isfinite(country_sd) else np.nan,
                    "n_obs": int((working["country_code"] == country).sum()),
                }
            )
    else:
        grp = working.groupby("country_code")["leniency_residual"]
        for country, val in grp.mean().items():
            n = int((working["country_code"] == country).sum())
            records.append(
                {
                    "jurisdiction_level": "Country",
                    "jurisdiction": country,
                    "effect": float(val),
                    "sd": country_sd,
                    "lower": float(val) - 1.96 * country_sd if np.isfinite(country_sd) else np.nan,
                    "upper": float(val) + 1.96 * country_sd if np.isfinite(country_sd) else np.nan,
                    "n_obs": n,
                }
            )

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["jurisdiction_level", "effect"], ascending=[True, False])
    return LeniencyResult(frame=frame, base_model=base_model, dpa_model=dpa_model, country_model=country_model)


__all__ = ["compute_leniency_index", "LeniencyResult"]
