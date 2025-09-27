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
    formula = _make_formula(outcome, fact_features)
    base_model = smf.ols(formula, data=data).fit()
    residuals = base_model.resid
    working = data.copy()
    working["leniency_residual"] = residuals

    random_slope_terms = list(random_slope_terms or [])
    re_formula = "1"
    if random_slope_terms:
        re_formula = "1 + " + " + ".join(random_slope_terms)

    dpa_model = smf.mixedlm(
        "leniency_residual ~ 1",
        working,
        groups=working["dpa_name_canonical"],
        re_formula=re_formula,
    ).fit(reml=True)

    country_model = smf.mixedlm(
        "leniency_residual ~ 1",
        working,
        groups=working["country_code"],
    ).fit(reml=True)

    dpa_var = float(dpa_model.cov_re.iloc[0, 0]) if dpa_model.cov_re.size else float("nan")
    dpa_sd = float(np.sqrt(dpa_var)) if np.isfinite(dpa_var) else float("nan")
    country_var = float(country_model.cov_re.iloc[0, 0]) if country_model.cov_re.size else float("nan")
    country_sd = float(np.sqrt(country_var)) if np.isfinite(country_var) else float("nan")

    records: list[dict[str, object]] = []
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

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["jurisdiction_level", "effect"], ascending=[True, False])
    return LeniencyResult(frame=frame, base_model=base_model, dpa_model=dpa_model, country_model=country_model)


__all__ = ["compute_leniency_index", "LeniencyResult"]
