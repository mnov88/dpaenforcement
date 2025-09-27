"""Statistical models for conditional disparity estimation."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _fit_with_cov_type(model, cluster_col: str | None, data: pd.DataFrame, **fit_kwargs):
    if cluster_col:
        return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]}, **fit_kwargs)
    return model.fit(**fit_kwargs)


def fit_logistic(
    data: pd.DataFrame,
    formula: str,
    weight_col: str | None = None,
    cluster_col: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    weights = data[weight_col] if weight_col else None
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial(), freq_weights=weights)
    return _fit_with_cov_type(model, cluster_col, data)


def fit_ols(
    data: pd.DataFrame,
    formula: str,
    weight_col: str | None = None,
    cluster_col: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    weights = data[weight_col] if weight_col else None
    model = smf.wls(formula=formula, data=data, weights=weights if weight_col else None)
    return _fit_with_cov_type(model, cluster_col, data)


def fit_mixed_effects(
    data: pd.DataFrame,
    formula: str,
    group_col: str,
    re_formula: str | None = None,
    vc_formula: Mapping[str, str] | None = None,
    weight_col: str | None = None,
) -> sm.regression.mixed_linear_model.MixedLMResults:
    weights = data[weight_col] if weight_col else None
    model = smf.mixedlm(formula, data, groups=data[group_col], re_formula=re_formula, vc_formula=vc_formula, weights=weights)
    return model.fit(method="lbfgs", reml=True)


def model_to_dict(result: Any) -> Dict[str, Any]:
    summary = {
        "params": result.params.to_dict(),
        "bse": result.bse.to_dict(),
        "pvalues": result.pvalues.to_dict(),
        "aic": getattr(result, "aic", np.nan),
        "bic": getattr(result, "bic", np.nan),
        "deviance": getattr(result, "deviance", np.nan),
        "nobs": float(getattr(result, "nobs", np.nan)),
    }
    if hasattr(result, "random_effects"):
        summary["random_effects"] = {k: v.tolist() for k, v in result.random_effects.items()}
    if hasattr(result, "cov_re"):
        summary["cov_re"] = result.cov_re.tolist()
    return summary


def marginal_effects(result: sm.regression.linear_model.RegressionResultsWrapper, at: Mapping[str, float] | None = None) -> pd.DataFrame:
    if not hasattr(result, "get_margeff"):
        raise TypeError("Model does not support marginal effects")
    margeff = result.get_margeff(at=at)
    frame = margeff.summary_frame()
    frame.index.name = "term"
    return frame.reset_index()


__all__ = ["fit_logistic", "fit_ols", "fit_mixed_effects", "model_to_dict", "marginal_effects"]
