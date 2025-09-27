"""Robustness and sensitivity checks."""
from __future__ import annotations

from typing import Dict, Mapping

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .config import FACTS_CONFIG
from .modeling import fit_logistic, fit_ols, model_to_dict

ScenarioResult = Dict[str, object]


def _inverse_mills_ratio(linear_pred: np.ndarray) -> np.ndarray:
    pdf = stats.norm.pdf(linear_pred)
    cdf = stats.norm.cdf(linear_pred)
    cdf = np.clip(cdf, 1e-9, 1 - 1e-9)
    return pdf / cdf


def _apply_selection_correction(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, object]:
    working = df.copy()
    working["turnover_observed"] = working["turnover_log1p"].notna().astype(int)
    formula = "turnover_observed ~ " + " + ".join(features)
    probit = smf.probit(formula, data=working).fit(disp=False)
    linear_pred = probit.predict(linear=True)
    working["turnover_control"] = _inverse_mills_ratio(linear_pred)
    return working, probit


def _apply_winsorization(df: pd.DataFrame, column: str, quantile: float) -> pd.DataFrame:
    upper = df[column].quantile(quantile)
    lower = df[column].quantile(1 - quantile)
    df = df.copy()
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def default_model_builder(
    data: pd.DataFrame,
    options: Mapping[str, object],
    logistic_formula: str,
    linear_formula: str,
    cluster_col: str = "dpa_name_canonical",
) -> Dict[str, object]:
    logistic = fit_logistic(data, logistic_formula, cluster_col=cluster_col)
    linear = fit_ols(data, linear_formula, cluster_col=cluster_col)
    return {"logistic": model_to_dict(logistic), "linear": model_to_dict(linear)}


def run_robustness_suite(
    data: pd.DataFrame,
    scenarios: Mapping[str, Mapping[str, object]],
    logistic_formula: str,
    linear_formula: str,
    fact_features: list[str],
    cluster_col: str = "dpa_name_canonical",
) -> Dict[str, ScenarioResult]:
    outputs: Dict[str, ScenarioResult] = {}
    for name, options in scenarios.items():
        working = data.copy()
        notes: list[str] = []
        if options.get("discussed_only"):
            mask = pd.Series(True, index=working.index)
            for prefix in FACTS_CONFIG.multi_value_prefixes:
                status_col = f"{prefix}_status"
                if status_col in working:
                    mask &= working[status_col].fillna("MISSING").eq("DISCUSSED")
            working = working.loc[mask]
            notes.append("Filtered to discussed-only records")
        if options.get("winsorize"):
            working = _apply_winsorization(working, "fine_log1p", options["winsorize"])
            notes.append(f"Winsorized fine_log1p at q={options['winsorize']}")
        if options.get("quantile"):
            quant_model = smf.quantreg(linear_formula, working).fit(q=options["quantile"])
            outputs[name] = {
                "type": "quantile",
                "quantile": options["quantile"],
                "params": quant_model.params.to_dict(),
                "nobs": float(quant_model.nobs),
                "notes": notes,
            }
            continue
        if options.get("selection") == "turnover":
            working, probit = _apply_selection_correction(working, fact_features)
            notes.append("Applied turnover control function")
        if options.get("weighting") == "country_year":
            counts = working.groupby("country_year").size()
            weights = 1.0 / counts
            working["robust_weight"] = working["country_year"].map(weights)
            notes.append("Applied country-year reweighting")
            weight_col = "robust_weight"
        else:
            weight_col = None
        logistic = fit_logistic(
            working,
            logistic_formula,
            weight_col=weight_col,
            cluster_col=cluster_col,
        )
        linear = fit_ols(
            working,
            linear_formula,
            weight_col=weight_col,
            cluster_col=cluster_col,
        )
        outputs[name] = {
            "type": "glm_ols",
            "logistic": model_to_dict(logistic),
            "linear": model_to_dict(linear),
            "notes": notes,
            "nobs": float(len(working)),
        }
    return outputs


__all__ = ["run_robustness_suite", "default_model_builder"]
