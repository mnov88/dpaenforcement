"""Hybrid exact/Gower matching utilities."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import MatchingSpec


def _format_stratum_key(values: Sequence[object]) -> str:
    parts = ["NA" if pd.isna(v) else str(v) for v in values]
    return "|".join(parts)


def _gower_distance_for_row(
    row_idx: int,
    candidate_idx: Iterable[int],
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    ranges: pd.Series,
) -> pd.Series:
    distances: dict[int, float] = {}
    row_num = numeric.loc[row_idx] if not numeric.empty else pd.Series(dtype=float)
    row_cat = categorical.loc[row_idx] if not categorical.empty else pd.Series(dtype="string")
    for cand in candidate_idx:
        cand_num = numeric.loc[cand] if not numeric.empty else pd.Series(dtype=float)
        cand_cat = categorical.loc[cand] if not categorical.empty else pd.Series(dtype="string")
        contrib = 0.0
        denom = 0
        if not numeric.empty:
            for col, rng in ranges.items():
                if col not in row_num.index:
                    continue
                r_val = row_num[col]
                c_val = cand_num[col]
                if pd.isna(r_val) or pd.isna(c_val):
                    continue
                if rng == 0 or pd.isna(rng):
                    continue
                contrib += abs(float(r_val) - float(c_val)) / float(rng)
                denom += 1
        if not categorical.empty:
            for col in categorical.columns:
                r_val = row_cat[col]
                c_val = cand_cat[col]
                if pd.isna(r_val) or pd.isna(c_val):
                    continue
                contrib += 0.0 if r_val == c_val else 1.0
                denom += 1
        distances[cand] = np.nan if denom == 0 else contrib / denom
    return pd.Series(distances)


def _prepare_numeric(block: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=block.index)
    numeric = block.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric


def _prepare_categorical(block: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=block.index)
    cat = block.loc[:, list(columns)].copy()
    for col in cat.columns:
        cat[col] = cat[col].astype("string")
    return cat


def perform_matching(
    data: pd.DataFrame,
    spec: MatchingSpec,
    within_country: bool,
    id_col: str = "decision_id",
    match_type: str = "within",
) -> pd.DataFrame:
    """Run matching and return a tidy edge list."""

    results: list[dict[str, object]] = []
    if data.empty:
        return pd.DataFrame(columns=["source_id", "target_id", "distance", "weight", "match_type", "stratum_key"])

    numeric = _prepare_numeric(data, spec.gower_numeric)
    categorical = _prepare_categorical(data, spec.gower_categorical)
    ranges = numeric.max(skipna=True) - numeric.min(skipna=True)
    ranges = ranges.replace({0: 1}).fillna(1)

    grouped = data.groupby(list(spec.exact_features), dropna=False)
    for key, block in grouped:
        if len(block) < spec.min_group_size:
            continue
        stratum_key = _format_stratum_key(key if isinstance(key, tuple) else (key,))
        block_numeric = numeric.loc[block.index]
        block_categorical = categorical.loc[block.index]
        block_ranges = block_numeric.max(skipna=True) - block_numeric.min(skipna=True)
        block_ranges = block_ranges.replace({0: 1}).fillna(1)
        for row_idx, row in block.iterrows():
            if within_country:
                mask = block["country_code"].astype(str) == str(row["country_code"])
                candidates = block.loc[mask & (block.index != row_idx)]
            else:
                candidates = block.loc[block["country_code"].astype(str) != str(row["country_code"])]
            if candidates.empty:
                continue
            distances = _gower_distance_for_row(
                row_idx,
                candidates.index,
                block_numeric,
                block_categorical,
                block_ranges,
            )
            distances = distances.dropna()
            if distances.empty:
                continue
            within_caliper = distances[distances <= spec.caliper]
            if within_caliper.empty:
                continue
            top = within_caliper.sort_values().head(spec.neighbours)
            weight = 1.0 / len(top)
            for rank, (cand_idx, dist) in enumerate(top.items(), start=1):
                results.append(
                    {
                        "source_id": row[id_col],
                        "target_id": block.loc[cand_idx, id_col],
                        "distance": float(dist),
                        "weight": weight,
                        "match_rank": rank,
                        "match_type": match_type,
                        "stratum_key": stratum_key,
                        "source_country": row["country_code"],
                        "target_country": block.loc[cand_idx, "country_code"],
                    }
                )
    return pd.DataFrame(results)


__all__ = ["perform_matching"]
