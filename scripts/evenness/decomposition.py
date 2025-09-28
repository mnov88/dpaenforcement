"""Disparity decomposition tools."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.oaxaca import OaxacaBlinder


def _expand_by_weights(frame: pd.DataFrame, weights: pd.Series, target: int = 5000) -> pd.DataFrame:
    """Approximate analytic weights by repeating observations.

    Statsmodels 0.14 removed the explicit ``weights`` argument from
    :class:`~statsmodels.stats.oaxaca.OaxacaBlinder`. To preserve the
    weighting behaviour used in earlier runs we up-weight observations by
    repeating them proportional to their analytic weight. The repetitions are
    capped by ``target`` to avoid exploding the sample size.
    """

    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    positive = weights > 0
    if not positive.any():
        return frame

    frame = frame.loc[positive].copy()
    weights = weights.loc[frame.index]
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return frame

    target = max(int(target), len(frame))
    scaled = (weights / total * target).round().astype(int)
    scaled[scaled < 1] = 1
    expanded_index = np.repeat(frame.index.to_numpy(), scaled.to_numpy())
    return frame.loc[expanded_index].copy()


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
    numeric_features = list(dict.fromkeys(features))
    subset[outcome] = pd.to_numeric(subset[outcome], errors="coerce")
    for feature in numeric_features:
        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
    subset = subset.dropna(subset=numeric_features + [outcome])
    if subset.empty:
        return pd.DataFrame()

    weights = None
    if weight_col and weight_col in subset.columns:
        weights = pd.to_numeric(subset[weight_col], errors="coerce")
        mask = weights.notna() & (weights > 0)
        subset = subset.loc[mask].copy()
        weights = weights.loc[mask]
    n_original = len(subset)

    if weights is not None and not weights.empty:
        subset = _expand_by_weights(subset, weights)

    if subset.empty:
        return pd.DataFrame()

    indicator_col = "__oaxaca_group_indicator"
    exog = subset[numeric_features].copy()
    exog[indicator_col] = subset[group_col].astype(str).eq(str(group_a)).astype(int)

    try:
        model = OaxacaBlinder(
            subset[outcome],
            exog,
            indicator_col,
            hasconst=False,
        )
        results = model.two_fold()
    except Exception:
        return pd.DataFrame()

    params = getattr(results, "params", (float("nan"),) * 3)
    explained = float(params[1]) if len(params) > 1 else float("nan")
    unexplained = float(params[0]) if params else float("nan")
    overall = float(params[2]) if len(params) > 2 else explained + unexplained
    parts = {
        "explained": explained,
        "unexplained": unexplained,
        "overall": overall,
        "explained_se": float("nan"),
        "unexplained_se": float("nan"),
        "overall_se": float("nan"),
    }
    record = {
        **parts,
        "group_a": group_a,
        "group_b": group_b,
        "outcome": outcome,
        "n_obs": n_original,
    }
    return pd.DataFrame([record])


__all__ = ["run_oaxaca_blinder"]
