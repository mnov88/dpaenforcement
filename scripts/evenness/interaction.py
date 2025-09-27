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
    cluster_col: str | None = None,
) -> pd.DataFrame:
    base_model = smf.ols(base_formula, data=data).fit()
    records: list[dict[str, float]] = []
    for term in interaction_terms:
        formula = base_formula + f" + C(country_code):{term}"
        model = smf.ols(formula, data=data).fit()
        delta_aic = model.aic - base_model.aic
        lr_stat = 2 * (model.llf - base_model.llf)
        records.append(
            {
                "term": term,
                "delta_aic": delta_aic,
                "lr_stat": lr_stat,
                "pvalue": model.compare_lr_test(base_model)[1],
            }
        )
    return pd.DataFrame(records).sort_values("delta_aic")


__all__ = ["interaction_scan"]
