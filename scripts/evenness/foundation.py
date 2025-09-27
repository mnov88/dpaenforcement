"""Phase 1 foundations for the GDPR evenness project.

This module prepares the design matrices and "twin" constructions that
underpin the subsequent inferential work.  It follows the specification in the
research plan by:

* constructing facts-only design matrices (full sample and time-observed
  subset with inverse-probability weights for the latter);
* harmonising country codes and logging the applied mappings;
* implementing three definitions of comparable cases (rule-based "CEM"
  buckets, Gower nearest-neighbours, and risk-band groupings); and
* emitting balance/coverage diagnostics together with common-support plots.

All outputs are written to ``outputs/evenness`` using the locations declared in
``EvennessPaths``.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .config import FACTS_CONFIG, EvennessPaths


STATUS_VALUE_ORDER = (
    "DISCUSSED",
    "NOT_DISCUSSED",
    "NOT_APPLICABLE",
    "NOT_MENTIONED",
    "UNCLEAR",
    "MISSING",
)


@dataclass
class PhaseOneOutputs:
    """Container returned by :func:`run_phase_one` for convenience."""

    X_full: pd.DataFrame
    X_timeobs: pd.DataFrame
    twins_cem: pd.DataFrame
    twins_gower_within: pd.DataFrame
    twins_gower_cross: pd.DataFrame
    twins_riskbands: pd.DataFrame
    balance: pd.DataFrame
    coverage: Mapping[str, pd.DataFrame]
    country_log: pd.DataFrame


def _slug(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .upper()
    )


def _harmonise_country(series: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Standardise country codes and produce a mapping log."""

    cleaned: list[str | pd.NA] = []
    log_records: list[dict[str, object]] = []
    known_overrides = {
        "COUNTRY OF THE DECIDING AUTHORITY: IRELAND (IE)": "IE",
        "UK": "GB",
    }
    valid_codes = {
        "AT",
        "BE",
        "BG",
        "CH",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "EU",
        "FI",
        "FR",
        "GB",
        "GR",
        "HR",
        "HU",
        "IE",
        "IS",
        "IT",
        "LI",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "NO",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
    }

    for original in series.fillna("").astype(str):
        raw = original.strip()
        upper = raw.upper()
        harmonised: str | pd.NA
        reason: str
        if upper == "" or upper in {"NOT_DISCUSSED", "NOT_APPLICABLE", "UNCLEAR"}:
            harmonised = pd.NA
            reason = upper or "MISSING"
        elif upper in known_overrides:
            harmonised = known_overrides[upper]
            reason = "OVERRIDE"
        elif len(upper) == 2 and upper.isalpha():
            harmonised = upper
            reason = "DIRECT"
        elif "(" in upper and ")" in upper:
            candidate = upper.split("(")[-1].split(")")[0].strip()
            harmonised = candidate if len(candidate) == 2 else pd.NA
            reason = "EMBEDDED" if harmonised is not pd.NA else "FAILED_EMBEDDED"
        else:
            harmonised = pd.NA
            reason = "UNMAPPED"
        if harmonised is not pd.NA and harmonised not in valid_codes:
            harmonised = pd.NA
            reason = "INVALID"
        cleaned.append(harmonised)
        log_records.append({
            "original": raw or pd.NA,
            "harmonised": harmonised,
            "status": reason,
        })
    log = pd.DataFrame(log_records).drop_duplicates().reset_index(drop=True)
    return pd.Series(cleaned, index=series.index, dtype="string"), log


def _status_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    status_col = f"{prefix}_status"
    coverage_col = f"{prefix}_coverage_status"
    indicators: dict[str, pd.Series] = {}
    series = df.get(status_col, pd.Series("MISSING", index=df.index, dtype="string"))
    series = series.fillna("MISSING").astype("string")
    for value in STATUS_VALUE_ORDER:
        col_name = f"{prefix}_status__{_slug(value)}"
        indicators[col_name] = series.eq(value).astype(int)
    if coverage_col in df:
        coverage = df[coverage_col].fillna("MISSING").astype("string")
        for value in sorted(coverage.unique()):
            col_name = f"{prefix}_coverage__{_slug(str(value))}"
            indicators[col_name] = coverage.eq(value).astype(int)
    return pd.DataFrame(indicators, index=df.index)


def _apply_status_gating(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    status_columns: list[str] = []
    working = df.copy()
    drop_mask = pd.Series(False, index=working.index)
    for prefix in FACTS_CONFIG.multi_value_prefixes:
        conflict_col = f"{prefix}_exclusivity_conflict"
        if conflict_col in working:
            drop_mask |= working[conflict_col].fillna(0).astype(bool)
    if drop_mask.any():
        working = working.loc[~drop_mask].copy()

    for prefix in FACTS_CONFIG.multi_value_prefixes:
        status_frame = _status_indicators(working, prefix)
        if not status_frame.empty:
            for col in status_frame.columns:
                working[col] = status_frame[col]
            status_columns.extend(status_frame.columns)

        status_col = f"{prefix}_status"
        indicators = [
            c
            for c in working.columns
            if c.startswith(f"{prefix}_")
            and c
            not in {
                status_col,
                f"{prefix}_coverage_status",
                f"{prefix}_exclusivity_conflict",
                f"{prefix}_known",
                f"{prefix}_unknown",
            }
        ]
        if not indicators:
            continue

        status = working.get(status_col, pd.Series("MISSING", index=working.index, dtype="string"))
        status = status.fillna("MISSING").astype("string")
        discussed_mask = status.eq("DISCUSSED")
        if discussed_mask.any():
            # ensure numeric and fill NaN with 0 for discussed rows
            working.loc[discussed_mask, indicators] = (
                working.loc[discussed_mask, indicators]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(float)
            )

        not_discussed = ~discussed_mask
        if not_discussed.any():
            explicit_cols = [c for c in indicators if c.endswith("NOT_MENTIONED") or c.endswith("NOT_APPLICABLE")]
            other_cols = [c for c in indicators if c not in explicit_cols]
            if other_cols:
                working.loc[not_discussed, other_cols] = 0.0
            for col in explicit_cols:
                if col.endswith("NOT_APPLICABLE"):
                    working.loc[:, col] = 0.0
                    working.loc[status.eq("NOT_APPLICABLE"), col] = 1.0
                elif col.endswith("NOT_MENTIONED"):
                    working.loc[:, col] = working.loc[:, col].fillna(0.0)
                    working.loc[status.eq("NOT_DISCUSSED"), col] = 1.0
            working.loc[not_discussed, indicators] = working.loc[not_discussed, indicators].fillna(0.0)

        working[indicators] = working[indicators].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return working, status_columns


def _country_year_weights(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("country_year").size()
    global_mean = counts.mean()
    weights = df["country_year"].map(counts).replace(0, np.nan)
    reweighted = global_mean / weights
    reweighted = reweighted.fillna(1.0)
    return reweighted


def _collect_fact_features(df: pd.DataFrame, status_cols: Sequence[str]) -> tuple[list[str], list[str]]:
    indicator_cols: list[str] = []
    for prefix in FACTS_CONFIG.multi_value_prefixes:
        indicator_cols.extend(
            [
                c
                for c in df.columns
                if c.startswith(f"{prefix}_")
                and c
                not in {
                    f"{prefix}_status",
                    f"{prefix}_coverage_status",
                    f"{prefix}_exclusivity_conflict",
                    f"{prefix}_known",
                    f"{prefix}_unknown",
                }
            ]
        )
    indicator_cols = sorted(dict.fromkeys(indicator_cols + list(status_cols)))

    numeric_cols = [
        col
        for col in (
            "n_principles_discussed",
            "n_principles_violated",
            "n_corrective_measures",
            "severity_measures_present",
            "remedy_only_case",
            "days_since_gdpr",
        )
        if col in df.columns
    ]
    return indicator_cols, numeric_cols


def _multi_signature(df: pd.DataFrame, prefix: str, none_label: str = "NONE") -> pd.Series:
    indicator_cols = [
        c
        for c in df.columns
        if c.startswith(f"{prefix}_")
        and not c.endswith("_status")
        and not c.endswith("_coverage_status")
        and not c.endswith("_exclusivity_conflict")
        and not c.endswith("_known")
        and not c.endswith("_unknown")
    ]
    if not indicator_cols:
        return pd.Series([none_label] * len(df), index=df.index, dtype="string")

    def encode(row: pd.Series) -> str:
        active = [
            col.split(f"{prefix}_", 1)[1]
            for col in indicator_cols
            if pd.to_numeric(row.get(col, 0), errors="coerce") == 1
        ]
        if not active:
            return none_label
        return "|".join(sorted(active))

    return df[indicator_cols].apply(encode, axis=1).astype("string")


def _design_matrix(df: pd.DataFrame, categorical: Sequence[str], indicator_cols: Sequence[str], numeric_cols: Sequence[str]) -> pd.DataFrame:
    working = df.copy()
    for col in categorical:
        if col in working.columns:
            working[col] = working[col].fillna("MISSING").astype("string")
    categorical = [col for col in categorical if col in working.columns]
    dummies = pd.get_dummies(working[categorical], prefix=[_slug(c) for c in categorical], dummy_na=False) if categorical else pd.DataFrame(index=working.index)
    indicators = working[list(indicator_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0) if indicator_cols else pd.DataFrame(index=working.index)
    numeric = working[list(numeric_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0) if numeric_cols else pd.DataFrame(index=working.index)
    matrix = pd.concat([indicators, numeric, dummies], axis=1)
    matrix.index = working.index
    return matrix


def _fit_time_ipw(df: pd.DataFrame, features: pd.DataFrame, mask: pd.Series) -> pd.Series:
    if mask.mean() in (0, 1):
        return pd.Series(1.0, index=df.index)
    X = features.copy()
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_scaled, mask.astype(int))
    probs = model.predict_proba(X_scaled)[:, 1]
    probs = np.clip(probs, 0.01, 0.99)
    ipw = pd.Series(1.0 / probs, index=df.index)
    ipw.loc[~mask] = np.nan
    return ipw


def _cem_twins(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "breach_case",
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
        "q25_sensitive_data_ARTICLE_10_CRIMINAL",
        "q46_vuln_CHILDREN",
        "decision_year_bucket",
        "organization_type",
        "organization_size_tier",
        "isic_section",
        "n_principles_discussed_bin",
        "n_principles_violated_bin",
        "n_corrective_measures_bin",
        "q21_signature",
        "q47_signature",
        "q25_signature",
        "q46_signature",
        "remedy_only_case",
        "q21_breach_types_status",
        "q21_breach_types_coverage_status",
        "q25_sensitive_data_status",
        "q25_sensitive_data_coverage_status",
        "q46_vuln_status",
        "q47_remedial_status",
    ]
    for col in group_cols:
        if col not in df.columns:
            df[col] = "MISSING"
        df[col] = df[col].fillna("MISSING").astype("string")
    grouped = df.groupby(group_cols, dropna=False)
    records: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for stratum_id, (key, block) in enumerate(grouped, start=1):
        size = len(block)
        countries = block["country_code"].dropna().unique()
        if size < 2 or len(countries) < 2:
            continue
        for _, row in block.iterrows():
            records.append(
                {
                    "decision_id": row["decision_id"],
                    "stratum_id": stratum_id,
                    "stratum_size": size,
                    "country_code": row["country_code"],
                    "weight": 1.0 / size,
                }
            )
        coverage.append(
            {
                "stratum_id": stratum_id,
                "n_cases": size,
                "n_countries": len(countries),
            }
        )
    twins = pd.DataFrame(records)
    coverage_df = pd.DataFrame(coverage)
    return twins, coverage_df


def _generate_cem_support_plot(twins: pd.DataFrame, path: Path) -> None:
    if twins.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(twins["stratum_size"], bins=20, color="#2a9d8f", edgecolor="black")
    plt.title("CEM twin stratum sizes")
    plt.xlabel("Stratum size")
    plt.ylabel("Number of cases")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _gower_distance(row: pd.Series, candidates: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.Series:
    distances: dict[int, float] = {}
    num_ranges = candidates[numeric_cols].max(skipna=True) - candidates[numeric_cols].min(skipna=True)
    num_ranges = num_ranges.replace({0: 1}).fillna(1)
    for idx, candidate in candidates.iterrows():
        total = 0.0
        denom = 0
        for col in numeric_cols:
            r = row[col]
            c = candidate[col]
            if pd.isna(r) or pd.isna(c):
                continue
            rng = num_ranges[col]
            if rng == 0 or pd.isna(rng):
                continue
            total += abs(float(r) - float(c)) / float(rng)
            denom += 1
        for col in categorical_cols:
            r = row[col]
            c = candidate[col]
            if pd.isna(r) or pd.isna(c):
                continue
            total += 0.0 if r == c else 1.0
            denom += 1
        distances[idx] = np.nan if denom == 0 else total / denom
    return pd.Series(distances)


def _build_near_twins(
    df: pd.DataFrame,
    indicator_cols: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    within_country: bool,
    caliper: float,
) -> pd.DataFrame:
    frame = df.copy()
    numeric_cols = [c for c in sorted(dict.fromkeys(list(numeric_cols) + list(indicator_cols))) if c in frame.columns]
    categorical_cols = [c for c in categorical_cols if c in frame.columns]
    for col in categorical_cols:
        frame[col] = frame[col].fillna("MISSING").astype("string")
    numeric_values = (
        frame.loc[:, numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    hard_match_cols = [
        col
        for col in numeric_cols
        if "_status__" in col
        or "_coverage__" in col
        or col.endswith("NOT_MENTIONED")
        or col.endswith("NOT_APPLICABLE")
    ]

    records: list[dict[str, object]] = []
    grouped = frame.groupby(["breach_case", "decision_year_bucket"], dropna=False)
    for _, block in grouped:
        if len(block) < 2:
            continue
        num = numeric_values.loc[block.index]
        if categorical_cols:
            dummies = pd.get_dummies(block[categorical_cols])
        else:
            dummies = pd.DataFrame(index=block.index)
        features = pd.concat([num, dummies], axis=1).fillna(0.0).astype(float)
        if features.empty:
            continue
        min_vals = features.min()
        ranges = features.max() - min_vals
        ranges = ranges.replace({0: 1}).fillna(1)
        normalised = (features - min_vals) / ranges
        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(normalised.values, metric="cityblock") / normalised.shape[1]
        countries = block["country_code"].fillna("MISSING").astype(str).values
        decision_ids = block["decision_id"].values
        status_block = (
            numeric_values.loc[block.index, hard_match_cols]
            if hard_match_cols
            else None
        )
        for i, (idx, row) in enumerate(block.iterrows()):
            row_country = countries[i]
            if within_country:
                mask = (countries == row_country) & (np.arange(len(block)) != i)
            else:
                mask = countries != row_country
            if status_block is not None and not status_block.empty:
                status_row = status_block.iloc[i]
                status_match = (status_block == status_row).all(axis=1).values
                mask &= status_match
            candidate_idx = np.where(mask)[0]
            if candidate_idx.size == 0:
                continue
            dist_values = distances[i, candidate_idx]
            within_caliper = dist_values <= caliper
            if not within_caliper.any():
                continue
            selected = candidate_idx[within_caliper]
            order = dist_values[within_caliper].argsort()[:5]
            chosen = selected[order]
            weights = 1.0 / len(chosen)
            for rank, j in enumerate(chosen, start=1):
                records.append(
                    {
                        "source_id": decision_ids[i],
                        "target_id": decision_ids[j],
                        "distance": float(distances[i, j]),
                        "weight": weights,
                        "match_rank": rank,
                        "source_country": countries[i],
                        "target_country": countries[j],
                        "within_country": within_country,
                    }
                )
    return pd.DataFrame(records)


def _generate_distance_plot(matches: pd.DataFrame, path: Path) -> None:
    if matches.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(matches["distance"], bins=20, color="#264653", edgecolor="black")
    plt.xlabel("Gower distance")
    plt.ylabel("Matches")
    plt.title("Near-twin distance distribution")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _train_risk_models(df: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    outcomes = df["fine_positive"].astype(int)
    classifier = LogisticRegression(max_iter=500, solver="liblinear")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        classifier.fit(features, outcomes)
    probas = pd.Series(classifier.predict_proba(features)[:, 1], index=df.index)

    mask = df["fine_log1p"].notna()
    if mask.sum() >= 10:
        reg = Ridge(alpha=1.0)
        reg.fit(features.loc[mask], df.loc[mask, "fine_log1p"].astype(float))
        preds = pd.Series(reg.predict(features), index=df.index)
    else:
        preds = pd.Series(np.nan, index=df.index)
    return probas, preds


def _build_risk_bands(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    probas, log_preds = _train_risk_models(df, features)
    ventiles = pd.qcut(probas.rank(method="first"), 20, labels=False) + 1
    risk = pd.DataFrame(
        {
            "decision_id": df["decision_id"],
            "pred_fine_positive": probas,
            "pred_log_fine": log_preds,
            "risk_ventile": ventiles,
        }
    )
    return risk


def _risk_support_plot(risk: pd.DataFrame, path: Path) -> None:
    if risk.empty:
        return
    plt.figure(figsize=(6, 4))
    counts = risk.groupby("risk_ventile").size()
    plt.bar(counts.index.astype(int), counts.values, color="#e76f51")
    plt.xlabel("Risk ventile")
    plt.ylabel("Cases")
    plt.title("Risk band coverage")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _cem_pairs(twins: pd.DataFrame) -> pd.DataFrame:
    pairs: list[dict[str, object]] = []
    for stratum_id, block in twins.groupby("stratum_id"):
        for (i_idx, i_row), (j_idx, j_row) in combinations(block.iterrows(), 2):
            raw_source = i_row.get("country_code", "MISSING")
            raw_target = j_row.get("country_code", "MISSING")
            source_country = "MISSING" if pd.isna(raw_source) or raw_source == "" else str(raw_source)
            target_country = "MISSING" if pd.isna(raw_target) or raw_target == "" else str(raw_target)
            if source_country == target_country:
                continue
            pairs.append(
                {
                    "source_id": i_row["decision_id"],
                    "target_id": j_row["decision_id"],
                    "weight": min(i_row["weight"], j_row["weight"]),
                    "stratum_id": stratum_id,
                    "source_country": source_country,
                    "target_country": target_country,
                }
            )
    return pd.DataFrame(pairs)


def _compute_smd(base: pd.DataFrame, compare: pd.DataFrame, features: Iterable[str], weight_col: str = "weight") -> pd.DataFrame:
    records: list[dict[str, object]] = []
    weights = compare.get(weight_col, pd.Series(1.0, index=compare.index))
    for feature in features:
        if feature not in base.columns or feature not in compare.columns:
            continue
        base_vals = pd.to_numeric(base[feature], errors="coerce")
        comp_vals = pd.to_numeric(compare[feature], errors="coerce")
        if base_vals.isna().all() or comp_vals.isna().all():
            continue
        mean_base = base_vals.mean()
        mean_comp = np.average(comp_vals.fillna(comp_vals.mean()), weights=weights.reindex(compare.index, fill_value=1.0))
        var_base = base_vals.var(ddof=0)
        var_comp = np.average((comp_vals - mean_comp) ** 2, weights=weights.reindex(compare.index, fill_value=1.0))
        denom = np.sqrt((var_base + var_comp) / 2) if var_base + var_comp > 0 else np.nan
        if denom is None or denom == 0 or np.isnan(denom):
            continue
        records.append({"feature": feature, "smd": (mean_comp - mean_base) / denom})
    return pd.DataFrame(records)


def _matched_balance(pairs: pd.DataFrame, feature_matrix: pd.DataFrame, features: Sequence[str], twin_type: str) -> list[dict[str, object]]:
    if pairs.empty:
        return []
    merged = pairs.merge(
        feature_matrix.add_prefix("source_"),
        left_on="source_id",
        right_on="source_decision_id",
        how="left",
    ).merge(
        feature_matrix.add_prefix("target_"),
        left_on="target_id",
        right_on="target_decision_id",
        how="left",
    )
    results: list[dict[str, object]] = []
    weights = pairs.get("weight", pd.Series(1.0, index=pairs.index)).reindex(merged.index, fill_value=1.0)
    for feature in features:
        s_col = f"source_{feature}"
        t_col = f"target_{feature}"
        if s_col not in merged or t_col not in merged:
            continue
        source = pd.to_numeric(merged[s_col], errors="coerce")
        target = pd.to_numeric(merged[t_col], errors="coerce")
        if source.isna().all() or target.isna().all():
            continue
        mean_s = np.average(source.fillna(source.mean()), weights=weights)
        mean_t = np.average(target.fillna(target.mean()), weights=weights)
        var_s = np.average((source - mean_s) ** 2, weights=weights)
        var_t = np.average((target - mean_t) ** 2, weights=weights)
        denom = np.sqrt((var_s + var_t) / 2) if (var_s + var_t) > 0 else np.nan
        if denom is None or np.isnan(denom) or denom == 0:
            continue
        results.append({"feature": feature, "smd": (mean_s - mean_t) / denom, "twin_type": twin_type})
    return results


def run_phase_one(df: pd.DataFrame, paths: EvennessPaths | None = None) -> PhaseOneOutputs:
    paths = paths or EvennessPaths()
    paths.ensure()

    harmonised, country_log = _harmonise_country(df["country_code"])
    df = df.copy()
    df["country_code"] = harmonised

    df, status_cols = _apply_status_gating(df)
    indicator_cols, numeric_cols = _collect_fact_features(df, status_cols)
    indicator_cols = [c for c in indicator_cols if not c.startswith("q53_powers_")]
    if "remedy_only_case" in df.columns:
        df["remedy_only_case"] = df["remedy_only_case"].fillna(False).astype(bool).map({True: "YES", False: "NO"}).astype("string")
    for col, name in [
        ("n_principles_discussed", "n_principles_discussed_bin"),
        ("n_principles_violated", "n_principles_violated_bin"),
        ("n_corrective_measures", "n_corrective_measures_bin"),
    ]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
            bins = [-1, 0, 2, 4, np.inf]
            labels = ["0", "1-2", "3-4", "5+"]
            df[name] = pd.cut(numeric, bins=bins, labels=labels).astype("string").fillna("0")
    df["q21_signature"] = _multi_signature(df, "q21_breach_types")
    df["q47_signature"] = _multi_signature(df, "q47_remedial")
    df["q25_signature"] = _multi_signature(df, "q25_sensitive_data")
    df["q46_signature"] = _multi_signature(df, "q46_vuln")
    categorical_cols = [
        "breach_case",
        "organization_size_tier",
        "organization_type",
        "case_origin",
        "decision_year_bucket",
        "isic_section",
        "n_principles_discussed_bin",
        "n_principles_violated_bin",
        "n_corrective_measures_bin",
        "q21_signature",
        "q47_signature",
        "q25_signature",
        "q46_signature",
        "remedy_only_case",
    ]

    df["country_year_weight"] = _country_year_weights(df)
    df["time_observed"] = df["decision_year"].notna() & df["decision_quarter"].notna()

    X_design = _design_matrix(df, categorical_cols, indicator_cols, numeric_cols)
    ipw = _fit_time_ipw(df, X_design, df["time_observed"])

    features_to_keep = [
        "decision_id",
        "country_code",
        "dpa_name_canonical",
        "decision_year",
        "decision_quarter",
        "days_since_gdpr",
        "country_year",
        "country_year_weight",
        "time_observed",
    ]
    base_cols = [col for col in features_to_keep if col in df.columns]
    outcomes = [col for col in FACTS_CONFIG.outcome_columns if col in df.columns]
    feature_matrix = pd.concat([df[base_cols + outcomes], X_design], axis=1)
    feature_matrix = feature_matrix.loc[:, ~feature_matrix.columns.duplicated()]
    feature_matrix.to_parquet(paths.x_full, index=False)

    timeobs = feature_matrix.loc[df["time_observed"]].copy()
    timeobs["ipw_time_observed"] = ipw.loc[timeobs.index]
    timeobs.to_parquet(paths.x_timeobs, index=False)

    cem_twins, cem_coverage = _cem_twins(df)
    cem_twins.to_parquet(paths.twins_cem, index=False)
    cem_coverage.to_csv(paths.coverage_dir / "cem_coverage.csv", index=False)
    _generate_cem_support_plot(cem_twins, paths.support_dir / "cem_support.png")

    gower_within = _build_near_twins(
        df,
        indicator_cols,
        numeric_cols,
        categorical_cols,
        within_country=True,
        caliper=0.1,
    )
    gower_cross = _build_near_twins(
        df,
        indicator_cols,
        numeric_cols,
        categorical_cols,
        within_country=False,
        caliper=0.03,
    )
    gower_within.to_parquet(paths.twins_gower_within, index=False)
    gower_cross.to_parquet(paths.twins_gower_cross, index=False)
    _generate_distance_plot(gower_within, paths.support_dir / "gower_within_support.png")
    _generate_distance_plot(gower_cross, paths.support_dir / "gower_cross_support.png")

    risk_bands = _build_risk_bands(df, X_design)
    risk_bands.to_parquet(paths.twins_riskbands, index=False)
    _risk_support_plot(risk_bands, paths.support_dir / "risk_band_support.png")

    coverage_tables = {
        "cem": cem_coverage,
        "gower_within": gower_within.groupby("source_country").size().reset_index(name="n_matches"),
        "gower_cross": gower_cross.groupby("source_country").size().reset_index(name="n_matches"),
        "risk_bands": risk_bands.groupby(["risk_ventile"]).size().reset_index(name="n_cases"),
    }
    for name, table in coverage_tables.items():
        table.to_csv(paths.coverage_dir / f"{name}_coverage.csv", index=False)

    cem_pairs = _cem_pairs(cem_twins)
    balance_frames = []
    cem_feature_set = indicator_cols + [
        "n_principles_discussed_bin",
        "n_principles_violated_bin",
        "n_corrective_measures_bin",
        "q21_signature",
        "q47_signature",
        "q25_signature",
        "q46_signature",
        "remedy_only_case",
    ]
    balance_frames.extend(_matched_balance(cem_pairs, feature_matrix, cem_feature_set, "cem"))
    balance_frames.extend(_matched_balance(gower_within, feature_matrix, indicator_cols + numeric_cols, "gower_within"))
    balance_frames.extend(_matched_balance(gower_cross, feature_matrix, indicator_cols + numeric_cols, "gower_cross"))

    risk_balance = _compute_smd(feature_matrix, feature_matrix.assign(weight=risk_bands["risk_ventile"].map(risk_bands["risk_ventile"].value_counts()).fillna(1.0)), indicator_cols + numeric_cols)
    if not risk_balance.empty:
        risk_balance["twin_type"] = "risk_bands"
        balance_frames.extend(risk_balance.to_dict("records"))

    balance_df = pd.DataFrame(balance_frames)
    balance_df.to_csv(paths.balance_csv, index=False)

    country_log.to_csv(paths.harmonization_log, index=False)

    return PhaseOneOutputs(
        X_full=feature_matrix,
        X_timeobs=timeobs,
        twins_cem=cem_twins,
        twins_gower_within=gower_within,
        twins_gower_cross=gower_cross,
        twins_riskbands=risk_bands,
        balance=balance_df,
        coverage=coverage_tables,
        country_log=country_log,
    )


__all__ = ["run_phase_one", "PhaseOneOutputs"]
