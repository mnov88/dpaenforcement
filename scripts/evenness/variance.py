"""Variance decomposition helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def intraclass_correlation(model) -> float:
    if not hasattr(model, "cov_re"):
        return float("nan")
    re_var = float(model.cov_re.iloc[0, 0]) if model.cov_re.size else 0.0
    resid = float(model.scale) if hasattr(model, "scale") else 0.0
    denom = re_var + resid
    return float(re_var / denom) if denom > 0 else float("nan")


def variance_summary(models: Dict[str, object]) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for name, model in models.items():
        re_var = float(model.cov_re.iloc[0, 0]) if hasattr(model, "cov_re") and model.cov_re.size else np.nan
        resid = float(model.scale) if hasattr(model, "scale") else np.nan
        icc = intraclass_correlation(model)
        records.append(
            {
                "component": name,
                "random_variance": re_var,
                "residual_variance": resid,
                "icc": icc,
            }
        )
    return pd.DataFrame(records)


__all__ = ["intraclass_correlation", "variance_summary"]
