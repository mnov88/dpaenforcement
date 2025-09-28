"""Jurisdiction-driver interaction diagnostics."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import statsmodels.formula.api as smf


def interaction_scan(
    data: pd.DataFrame,
    outcome: str,
    base_formula: str,
    interaction_terms: Iterable[str],
    group_field: str = "country_code",
) -> pd.DataFrame:
    if group_field not in data.columns:
        return pd.DataFrame(columns=["term", "delta_aic", "lr_stat", "pvalue", "group_field"])

    base_model = smf.ols(base_formula, data=data).fit()
    records: list[dict[str, float]] = []
    for term in interaction_terms:
        if term not in data.columns:
            continue
        formula = base_formula + f" + C({group_field}):{term}"
        try:
            model = smf.ols(formula, data=data).fit()
        except Exception:
            continue
        delta_aic = model.aic - base_model.aic
        lr_stat = 2 * (model.llf - base_model.llf)
        try:
            pvalue = model.compare_lr_test(base_model)[1]
        except Exception:
            pvalue = float("nan")
        records.append(
            {
                "term": term,
                "delta_aic": float(delta_aic),
                "lr_stat": float(lr_stat),
                "pvalue": float(pvalue),
                "group_field": group_field,
            }
        )
    frame = pd.DataFrame(records)
    return frame.sort_values("delta_aic") if not frame.empty else frame


__all__ = ["interaction_scan"]
