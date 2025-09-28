"""Phase 3 – explanation, policy levers, and robustness synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .config import (
    EvennessPaths,
    FACTS_CONFIG,
    LENIENCY_RANDOM_SLOPE_DRIVERS,
    ROBUSTNESS_SCENARIOS,
)
from .decomposition import run_oaxaca_blinder
from .interaction import interaction_scan
from .robustness import run_robustness_suite


@dataclass
class PhaseThreeOutputs:
    """Container for artefacts returned by :func:`run_phase_three`."""

    driver_leaderboard: pd.DataFrame
    country_interactions: pd.DataFrame
    dpa_interactions: pd.DataFrame
    decompositions: pd.DataFrame
    policy_estimates: pd.DataFrame
    rd_placebos: pd.DataFrame
    robustness_summary: pd.DataFrame
    randomization_inference: pd.DataFrame
    insights_report: Path
    playbook: Path
    environment_snapshot: Path


def _fact_columns(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "q21_breach_types_",
        "q25_sensitive_data_",
        "q46_vuln_",
        "q47_remedial_",
        "BREACH_CASE_",
        "CASE_ORIGIN_",
        "ORGANIZATION_SIZE_TIER_",
        "ORGANIZATION_TYPE_",
        "DECISION_YEAR_BUCKET_",
        "ISIC_SECTION_",
        "N_PRINCIPLES_DISCUSSED_BIN_",
        "N_PRINCIPLES_VIOLATED_BIN_",
        "N_CORRECTIVE_MEASURES_BIN_",
        "Q21_SIGNATURE_",
        "Q25_SIGNATURE_",
        "Q46_SIGNATURE_",
        "Q47_SIGNATURE_",
        "REMEDY_ONLY_CASE_",
    )
    columns: list[str] = []
    for col in df.columns:
        if col in {
            "decision_id",
            "country_code",
            "dpa_name_canonical",
            "country_year_weight",
            "time_observed",
            "ipw_time_observed",
        }:
            continue
        if col in FACTS_CONFIG.outcome_columns:
            continue
        if any(col.startswith(prefix) for prefix in prefixes):
            columns.append(col)
    for col in ["n_principles_violated", "n_corrective_measures", "days_since_gdpr"]:
        if col in df.columns:
            columns.append(col)
    return sorted(dict.fromkeys(columns))


def _driver_terms(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "q25_sensitive_data_",
        "q46_vuln_",
        "q47_remedial_",
        "q21_breach_types_",
        "BREACH_CASE_",
        "CASE_ORIGIN_",
        "ORGANIZATION_SIZE_TIER_",
        "ORGANIZATION_TYPE_",
    )
    candidates = set(LENIENCY_RANDOM_SLOPE_DRIVERS)
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in prefixes):
            candidates.add(col)
    for numeric in ("n_principles_violated", "n_corrective_measures"):
        if numeric in df.columns:
            candidates.add(numeric)
    return sorted(candidates)


def _build_formula(outcome: str, fact_terms: Sequence[str]) -> str:
    terms = list(fact_terms)
    terms.append("C(country_code)")
    terms.append("C(dpa_name_canonical)")
    rhs = "1"
    if terms:
        rhs = "1 + " + " + ".join(dict.fromkeys(terms))
    return f"{outcome} ~ {rhs}"


def _apply_fdr(df: pd.DataFrame, column: str = "pvalue", adj_column: str = "pvalue_fdr") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    mask = df[column].notna()
    if mask.sum() == 0:
        df[adj_column] = np.nan
        return df
    adjusted = np.full(len(df), np.nan)
    _, adj, _, _ = multipletests(df.loc[mask, column], method="fdr_bh")
    adjusted[mask] = adj
    df[adj_column] = adjusted
    return df


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _top_country_pairs(df: pd.DataFrame, limit: int = 5) -> list[tuple[str, str]]:
    if "country_code" not in df.columns:
        return []
    counts = df["country_code"].dropna().value_counts()
    if counts.size < 2:
        return []
    pairs: list[tuple[str, str]] = []
    for a_idx in range(min(len(counts), 6)):
        for b_idx in range(a_idx + 1, min(len(counts), 6)):
            a = counts.index[a_idx]
            b = counts.index[b_idx]
            pairs.append((str(a), str(b)))
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in unique:
            unique.append(pair)
        if len(unique) >= limit:
            break
    return unique


def _is_yes(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.upper()
    return values.isin({"YES", "Y", "TRUE", "1"})


def _late_indicator(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.upper()
    late_tokens = ("LATE", "AFTER", "OUTSIDE", "NO", "BEYOND")
    return values.str.contains("|".join(late_tokens)).astype(int)


def _centre_running_variable(values: pd.Series) -> tuple[pd.Series, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.replace({np.inf: np.nan, -np.inf: np.nan})
    if numeric.abs().median(skipna=True) is not None and numeric.abs().median(skipna=True) < 24:
        return numeric, 0.0
    return numeric - 72.0, 72.0


def _local_linear_rd(
    running: pd.Series,
    outcome: pd.Series,
    bandwidth: float,
    donut: float = 0.0,
) -> dict[str, float] | None:
    mask = running.notna() & outcome.notna()
    work = pd.DataFrame({"running": running[mask], "outcome": outcome[mask]})
    if work.empty:
        return None
    if donut > 0:
        work = work.loc[work["running"].abs() >= donut]
    work = work.loc[work["running"].abs() <= bandwidth]
    if work.empty or work["running"].nunique() < 3:
        return None
    work = work.copy()
    work["post"] = (work["running"] >= 0).astype(int)
    work["running_post"] = work["running"] * work["post"]
    design = sm.add_constant(work[["post", "running", "running_post"]], has_constant="add")
    weights = 1.0 - (work["running"].abs() / bandwidth)
    weights = weights.clip(lower=0.0)
    try:
        model = sm.WLS(work["outcome"], design, weights=weights).fit()
    except Exception:
        return None
    if "post" not in model.params:
        return None
    estimate = float(model.params["post"])
    std_err = float(model.bse["post"])
    return {
        "estimate": estimate,
        "std_err": std_err,
        "n_obs": float(model.nobs),
    }


def estimate_timing_effect(
    data: pd.DataFrame,
    outcomes: Sequence[str],
    bandwidth: float = 72.0,
    donut: float = 12.0,
    placebo_offsets: Sequence[float] = (24.0, 48.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "art33_required" not in data.columns or "art33_delay_amount" not in data.columns:
        return pd.DataFrame(), pd.DataFrame()
    required_mask = _is_yes(data["art33_required"])
    working = data.loc[required_mask].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    running, cutoff_used = _centre_running_variable(working["art33_delay_amount"])
    working["running"] = running
    working["late_indicator"] = _late_indicator(working.get("art33_submission_timing", pd.Series(index=working.index)))

    policy_records: list[dict[str, object]] = []
    placebo_records: list[dict[str, object]] = []

    first_stage = _local_linear_rd(working["running"], working["late_indicator"], bandwidth, donut)
    for outcome in outcomes:
        if outcome not in working.columns:
            continue
        rd_result = _local_linear_rd(working["running"], working[outcome], bandwidth, donut)
        if rd_result is None:
            continue
        record = {
            "lever": "art33_timing",
            "outcome": outcome,
            "result_type": "reduced_form",
            "estimate": rd_result["estimate"],
            "std_err": rd_result["std_err"],
            "ci_lower": rd_result["estimate"] - 1.96 * rd_result["std_err"],
            "ci_upper": rd_result["estimate"] + 1.96 * rd_result["std_err"],
            "bandwidth": bandwidth,
            "donut": donut,
            "cutoff": cutoff_used,
            "n_obs": rd_result["n_obs"],
        }
        if first_stage is not None and abs(first_stage["estimate"]) > 1e-6:
            tau = rd_result["estimate"] / first_stage["estimate"]
            var = (rd_result["std_err"] / first_stage["estimate"]) ** 2
            var += (
                rd_result["estimate"]
                * first_stage["std_err"]
                / (first_stage["estimate"] ** 2)
            ) ** 2
            record["late_first_stage"] = first_stage["estimate"]
            policy_records.append(record)
            policy_records.append(
                {
                    "lever": "art33_timing",
                    "outcome": outcome,
                    "result_type": "local_ate",
                    "estimate": tau,
                    "std_err": math.sqrt(var),
                    "ci_lower": tau - 1.96 * math.sqrt(var),
                    "ci_upper": tau + 1.96 * math.sqrt(var),
                    "bandwidth": bandwidth,
                    "donut": donut,
                    "cutoff": cutoff_used,
                    "n_obs": rd_result["n_obs"],
                    "late_first_stage": first_stage["estimate"],
                }
            )
        else:
            record["late_first_stage"] = np.nan
            policy_records.append(record)

        for offset in placebo_offsets:
            shifted = working["running"] - offset
            placebo = _local_linear_rd(shifted, working[outcome], bandwidth, donut)
            if placebo is not None:
                placebo_records.append(
                    {
                        "lever": "art33_timing",
                        "outcome": outcome,
                        "offset": offset,
                        "estimate": placebo["estimate"],
                        "std_err": placebo["std_err"],
                        "n_obs": placebo["n_obs"],
                    }
                )
            shifted = working["running"] + offset
            placebo = _local_linear_rd(shifted, working[outcome], bandwidth, donut)
            if placebo is not None:
                placebo_records.append(
                    {
                        "lever": "art33_timing",
                        "outcome": outcome,
                        "offset": -offset,
                        "estimate": placebo["estimate"],
                        "std_err": placebo["std_err"],
                        "n_obs": placebo["n_obs"],
                    }
                )

    return pd.DataFrame(policy_records), pd.DataFrame(placebo_records)


def _prepare_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame(index=df.index)
    features = df.loc[:, [col for col in feature_cols if col in df.columns]].copy()
    return features.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def estimate_notification_effect(
    data: pd.DataFrame,
    outcomes: Sequence[str],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    if "art34_required" not in data.columns or "subjects_notified" not in data.columns:
        return pd.DataFrame()
    mask = _is_yes(data["art34_required"])
    working = data.loc[mask].copy()
    if working.empty:
        return pd.DataFrame()
    treatment_raw = working["subjects_notified"]
    treated = treatment_raw.fillna("").astype(str).str.upper().isin({"YES", "Y", "TRUE", "1"}).astype(int)
    if treated.sum() < 2 or (len(treated) - treated.sum()) < 2:
        return pd.DataFrame()

    features = _prepare_features(working, feature_cols)
    design = sm.add_constant(features, has_constant="add")

    try:
        propensity_model = sm.Logit(treated, design).fit(disp=False)
    except Exception:
        return pd.DataFrame()
    propensity = propensity_model.predict(design)
    propensity = np.clip(propensity, 0.01, 0.99)

    records: list[dict[str, object]] = []
    for outcome in outcomes:
        if outcome not in working.columns:
            continue
        outcome_series = pd.to_numeric(working[outcome], errors="coerce")
        mask_valid = outcome_series.notna()
        if mask_valid.sum() < 4:
            continue
        y = outcome_series.loc[mask_valid]
        X = design.loc[mask_valid]
        t = treated.loc[mask_valid]
        p = propensity.loc[mask_valid]

        try:
            model_t = sm.OLS(y[t.astype(bool)], X.loc[t.astype(bool)]).fit()
            model_c = sm.OLS(y[~t.astype(bool)], X.loc[~t.astype(bool)]).fit()
        except Exception:
            continue

        mu1 = model_t.predict(X)
        mu0 = model_c.predict(X)
        aipw = mu1 - mu0 + (t / p) * (y - mu1) - ((1 - t) / (1 - p)) * (y - mu0)
        ate = float(aipw.mean())
        ate_se = float(aipw.std(ddof=1) / math.sqrt(len(aipw))) if len(aipw) > 1 else float("nan")
        att_mask = t.astype(bool)
        if att_mask.sum() > 1:
            att_values = aipw.loc[att_mask]
            att = float(att_values.mean())
            att_se = float(att_values.std(ddof=1) / math.sqrt(len(att_values)))
        else:
            att = float("nan")
            att_se = float("nan")
        records.append(
            {
                "lever": "subjects_notified",
                "outcome": outcome,
                "result_type": "ate",
                "estimate": ate,
                "std_err": ate_se,
                "ci_lower": ate - 1.96 * ate_se if np.isfinite(ate_se) else np.nan,
                "ci_upper": ate + 1.96 * ate_se if np.isfinite(ate_se) else np.nan,
                "overlap_min": float(p.min()),
                "overlap_max": float(p.max()),
                "effective_n": float(((p * (1 - p)).sum() ** 2) / ((p ** 2 * (1 - p) ** 2).sum()))
                if (p ** 2 * (1 - p) ** 2).sum() > 0
                else np.nan,
            }
        )
        records.append(
            {
                "lever": "subjects_notified",
                "outcome": outcome,
                "result_type": "att",
                "estimate": att,
                "std_err": att_se,
                "ci_lower": att - 1.96 * att_se if np.isfinite(att_se) else np.nan,
                "ci_upper": att + 1.96 * att_se if np.isfinite(att_se) else np.nan,
                "overlap_min": float(p.min()),
                "overlap_max": float(p.max()),
                "effective_n": float(((p * (1 - p)).sum() ** 2) / ((p ** 2 * (1 - p) ** 2).sum()))
                if (p ** 2 * (1 - p) ** 2).sum() > 0
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def _randomization_inference(
    cem_twins: pd.DataFrame,
    facts: pd.DataFrame,
    outcome: str,
    n_permutations: int = 200,
) -> dict[str, float] | None:
    if cem_twins.empty or outcome not in facts.columns:
        return None
    merged = cem_twins.merge(facts[["decision_id", "country_code", outcome]], on="decision_id", how="left")
    merged = merged.dropna(subset=[outcome, "country_code"])
    if merged.empty:
        return None

    def _statistic(frame: pd.DataFrame) -> float:
        total = 0.0
        for _, block in frame.groupby("stratum_id"):
            weights = block.get("weight", pd.Series(1.0, index=block.index))
            stratum_mean = np.average(block[outcome], weights=weights)
            for _, sub in block.groupby("country_code"):
                w = sub.get("weight", pd.Series(1.0, index=sub.index)).sum()
                mean = np.average(sub[outcome], weights=sub.get("weight", pd.Series(1.0, index=sub.index)))
                total += w * (mean - stratum_mean) ** 2
        return float(total)

    observed = _statistic(merged)
    perms: list[float] = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        shuffled = merged.copy()
        for stratum, block in merged.groupby("stratum_id"):
            shuffled.loc[block.index, "country_code"] = rng.permutation(block["country_code"].values)
        perms.append(_statistic(shuffled))
    perms_arr = np.array(perms)
    pvalue = float(((perms_arr >= observed).sum() + 1) / (len(perms_arr) + 1))
    return {
        "outcome": outcome,
        "observed_stat": observed,
        "perm_mean": float(perms_arr.mean()),
        "perm_std": float(perms_arr.std(ddof=1)),
        "pvalue": pvalue,
    }


def _summarise_robustness(results: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    key_params = [
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
        "q46_vuln_CHILDREN",
    ]
    records: list[dict[str, object]] = []
    for name, payload in results.items():
        record: dict[str, object] = {
            "scenario": name,
            "type": payload.get("type"),
            "nobs": payload.get("nobs"),
        }
        if payload.get("type") == "glm_ols":
            logistic = payload.get("logistic", {})
            linear = payload.get("linear", {})
            for param in key_params:
                record[f"logistic_{param}"] = logistic.get("params", {}).get(param)
                record[f"linear_{param}"] = linear.get("params", {}).get(param)
        elif payload.get("type") == "quantile":
            record["quantile"] = payload.get("quantile")
        record["notes"] = "; ".join(payload.get("notes", [])) if payload.get("notes") else ""
        records.append(record)
    return pd.DataFrame(records)


def _capture_environment(path: Path) -> None:
    lines: list[str] = [f"Generated: {datetime.utcnow().isoformat()}Z"]
    try:
        python_version = subprocess.run(
            ["python", "--version"], capture_output=True, text=True, check=True
        )
        lines.append(python_version.stdout.strip())
    except Exception:
        lines.append("python --version: unavailable")
    try:
        pip_list = subprocess.run(
            ["python", "-m", "pip", "list"], capture_output=True, text=True, check=True
        )
        lines.append("\n".join(["pip list:", pip_list.stdout.strip()]))
    except Exception:
        lines.append("pip list unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No data available._"
    preview = df.head(max_rows)
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return "```\n" + preview.to_string(index=False) + "\n```"


def _render_insights(
    driver_leaderboard: pd.DataFrame,
    decompositions: pd.DataFrame,
    policy_estimates: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    randomization: pd.DataFrame,
    path: Path,
) -> None:
    sections = [
        "# Phase 3 – Explanation & Policy Synthesis",
        "## Driver Attribution",
        _to_markdown(driver_leaderboard),
        "\n## Gap Decomposition",
        _to_markdown(decompositions),
        "\n## Policy Lever Estimates",
        _to_markdown(policy_estimates),
        "\n## Randomisation Inference",
        _to_markdown(randomization),
        "\n## Robustness Summary",
        _to_markdown(robustness_summary),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections))


def _render_playbook(
    driver_leaderboard: pd.DataFrame,
    policy_estimates: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    path: Path,
) -> None:
    actions: list[str] = []
    if not driver_leaderboard.empty:
        top = driver_leaderboard.nsmallest(5, "delta_aic")
        action_lines = [
            "### Priority Drivers",
            _to_markdown(top[["group_field", "term", "pvalue_fdr"]]),
        ]
        actions.append("\n".join(action_lines))
    if not policy_estimates.empty:
        leverage = policy_estimates.loc[
            policy_estimates["result_type"].isin(["local_ate", "ate"])
        ]
        actions.append("### Policy Effects\n" + _to_markdown(leverage))
    if not robustness_summary.empty:
        actions.append("### Robustness Diagnostics\n" + _to_markdown(robustness_summary))
    content = "# Uniform Treatment Playbook\n" + "\n\n".join(actions or ["_No actionable levers identified._"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _plot_policy_effects(policy_estimates: pd.DataFrame, path: Path) -> None:
    if policy_estimates.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    focus = policy_estimates.loc[
        policy_estimates["result_type"].isin(["local_ate", "ate"])
    ].copy()
    if focus.empty:
        return
    focus = focus.sort_values("estimate")
    plt.figure(figsize=(6, max(3, len(focus) * 0.4)))
    plt.errorbar(
        focus["estimate"],
        np.arange(len(focus)),
        xerr=1.96 * focus["std_err"],
        fmt="o",
        color="#1d3557",
    )
    plt.axvline(0, color="#e63946", linestyle="--", linewidth=1)
    plt.yticks(np.arange(len(focus)), focus["lever"] + " → " + focus["outcome"])
    plt.xlabel("Estimated effect (with 95% CI)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def run_phase_three(
    paths: EvennessPaths | None = None,
    outcome: str = "fine_log1p",
    decomposition_outcomes: Sequence[str] | None = None,
    policy_outcomes: Sequence[str] | None = None,
) -> PhaseThreeOutputs:
    paths = paths or EvennessPaths()
    paths.ensure()

    facts = pd.read_parquet(paths.x_full)
    fact_terms = _fact_columns(facts)
    driver_terms = _driver_terms(facts)
    decomposition_outcomes = decomposition_outcomes or ("fine_log1p", "enforcement_severity_index")
    policy_outcomes = policy_outcomes or ("fine_positive", "fine_log1p")

    base_formula = _build_formula(outcome, fact_terms)
    country_interactions = interaction_scan(
        facts,
        outcome=outcome,
        base_formula=base_formula,
        interaction_terms=driver_terms,
        group_field="country_code",
    )
    dpa_interactions = interaction_scan(
        facts,
        outcome=outcome,
        base_formula=base_formula,
        interaction_terms=driver_terms,
        group_field="dpa_name_canonical",
    )
    driver_leaderboard = pd.concat([country_interactions, dpa_interactions], ignore_index=True, sort=False)
    driver_leaderboard = _apply_fdr(driver_leaderboard)

    _write_dataframe(country_interactions, paths.interaction_country_csv)
    _write_dataframe(dpa_interactions, paths.interaction_dpa_csv)
    _write_dataframe(driver_leaderboard, paths.driver_leaderboard_csv)

    weight_col = "country_year_weight" if "country_year_weight" in facts.columns else None
    decomposition_frames: list[pd.DataFrame] = []
    for group_a, group_b in _top_country_pairs(facts):
        for target_outcome in decomposition_outcomes:
            if target_outcome not in facts.columns:
                continue
            frame = run_oaxaca_blinder(
                facts,
                outcome=target_outcome,
                group_col="country_code",
                group_a=group_a,
                group_b=group_b,
                features=fact_terms,
                weight_col=weight_col,
            )
            if not frame.empty:
                frame["group_a"] = group_a
                frame["group_b"] = group_b
                decomposition_frames.append(frame)
    decompositions = pd.concat(decomposition_frames, ignore_index=True) if decomposition_frames else pd.DataFrame()
    _write_dataframe(decompositions, paths.decomposition_summary_csv)

    timing_effects, placebo_checks = estimate_timing_effect(facts, policy_outcomes)
    notification_effects = estimate_notification_effect(facts, policy_outcomes, fact_terms)
    policy_estimates = pd.concat(
        [frame for frame in (timing_effects, notification_effects) if not frame.empty],
        ignore_index=True,
    )
    _write_dataframe(policy_estimates, paths.policy_estimates_csv)
    _write_dataframe(placebo_checks, paths.policy_placebo_csv)

    cem_twins = pd.read_parquet(paths.twins_cem)
    randomization_records: list[dict[str, float]] = []
    for target_outcome in policy_outcomes:
        result = _randomization_inference(cem_twins, facts, target_outcome)
        if result:
            randomization_records.append(result)
    randomization_df = pd.DataFrame(randomization_records)
    _write_dataframe(randomization_df, paths.randomization_csv)

    robustness_payload = run_robustness_suite(
        facts,
        ROBUSTNESS_SCENARIOS,
        _build_formula("fine_positive", fact_terms),
        _build_formula("fine_log1p", fact_terms),
        fact_terms,
    )
    robustness_summary = _summarise_robustness(robustness_payload)
    _write_dataframe(robustness_summary, paths.robustness_summary_csv)

    _plot_policy_effects(policy_estimates, paths.policy_plot)

    _render_insights(driver_leaderboard, decompositions, policy_estimates, robustness_summary, randomization_df, paths.insights_report)
    _render_playbook(driver_leaderboard, policy_estimates, robustness_summary, paths.playbook_report)
    _capture_environment(paths.environment_snapshot)

    return PhaseThreeOutputs(
        driver_leaderboard=driver_leaderboard,
        country_interactions=country_interactions,
        dpa_interactions=dpa_interactions,
        decompositions=decompositions,
        policy_estimates=policy_estimates,
        rd_placebos=placebo_checks,
        robustness_summary=robustness_summary,
        randomization_inference=randomization_df,
        insights_report=paths.insights_report,
        playbook=paths.playbook_report,
        environment_snapshot=paths.environment_snapshot,
    )


__all__ = [
    "PhaseThreeOutputs",
    "estimate_timing_effect",
    "estimate_notification_effect",
    "run_phase_three",
]
