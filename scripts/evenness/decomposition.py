"""Disparity decomposition tools."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from statsmodels.stats.oaxaca import OaxacaBlinder


def run_oaxaca_blinder(
    data: pd.DataFrame,
    outcome: str,
    group_col: str,
    group_a: str,
    group_b: str,
    features: Sequence[str],
    weight_col: str | None = None,
) -> pd.DataFrame:
    subset = data.loc[data[group_col].isin([group_a, group_b])].copy()
    subset = subset.dropna(subset=list(features) + [outcome])
    if subset.empty:
        return pd.DataFrame()
    weights = subset[weight_col] if weight_col else None
    model = OaxacaBlinder(
        subset[outcome],
        subset[list(features)],
        subset[group_col].eq(group_a).astype(int),
        hasconst=False,
        weights=weights,
    )
    results = model.blinder_oaxaca()
    parts = {
        "explained": getattr(results, "explained", float("nan")),
        "unexplained": getattr(results, "unexplained", float("nan")),
        "overall": getattr(results, "mean_group1", float("nan")) - getattr(results, "mean_group0", float("nan")),
        "explained_se": getattr(results, "explained_se", float("nan")),
        "unexplained_se": getattr(results, "unexplained_se", float("nan")),
        "overall_se": getattr(results, "overall_se", float("nan")),
    }
    record = {
        **parts,
        "group_a": group_a,
        "group_b": group_b,
        "outcome": outcome,
        "n_obs": len(subset),
    }
    return pd.DataFrame([record])


__all__ = ["run_oaxaca_blinder"]
