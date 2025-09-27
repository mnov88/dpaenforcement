from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "outputs"
WIDE_CSV = DATA_DIR / "cleaned_wide.csv"
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

GDPR_START = pd.Timestamp("2018-05-25", tz="UTC")
BOOTSTRAP_REPS = 20

STATUS_TOKENS_GENERIC = {
    "NOT_APPLICABLE",
    "NOT_MENTIONED",
    "NOT_DISCUSSED",
    "UNCLEAR",
}

@dataclass
class MultiParseResult:
    status: str
    tokens: List[str]
    conflict: int


def _split_tokens(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def parse_multiselect(raw: str | float, allowed_tokens: Iterable[str], *,
                      extra_status_tokens: Iterable[str] | None = None) -> MultiParseResult:
    """Parse a multi-select string into tokens, status and exclusivity flag."""
    status_tokens = set(STATUS_TOKENS_GENERIC)
    if extra_status_tokens:
        status_tokens.update(extra_status_tokens)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return MultiParseResult(status="MISSING", tokens=[], conflict=0)
    raw_str = str(raw).strip()
    if not raw_str:
        return MultiParseResult(status="MISSING", tokens=[], conflict=0)
    tokens = _split_tokens(raw_str)
    if not tokens:
        return MultiParseResult(status="MISSING", tokens=[], conflict=0)
    # If every token is a status marker, treat as status-only row
    if all(token in status_tokens for token in tokens):
        if len(tokens) == 1:
            return MultiParseResult(status=tokens[0], tokens=[], conflict=0)
        return MultiParseResult(status="MIXED_STATUS", tokens=[], conflict=1)
    conflict = int(any(token in status_tokens for token in tokens))
    clean_tokens = [token for token in tokens if token not in status_tokens]
    allowed = set(allowed_tokens)
    clean_tokens = [token for token in clean_tokens if token in allowed]
    return MultiParseResult(status="DISCUSSED", tokens=clean_tokens, conflict=conflict)


def harmonise_country(code: str | float) -> str | float:
    if isinstance(code, float) and np.isnan(code):
        return np.nan
    if not code:
        return code
    code_str = str(code).strip()
    if code_str.upper().startswith("COUNTRY OF THE DECIDING AUTHORITY: IRELAND"):
        return "IE"
    if code_str == "UK":
        return "GB"
    return code_str


def compute_country_weights(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    country_counts = df["country_code_clean"].value_counts(dropna=False)
    n_countries = country_counts[country_counts > 0].shape[0]
    total = country_counts.sum()
    weights = country_counts.apply(lambda c: total / (n_countries * c) if c > 0 else np.nan)
    country_weight = df["country_code_clean"].map(weights)

    observed = df.dropna(subset=["decision_year"])
    if not observed.empty:
        cy_counts = observed.groupby(["country_code_clean", "decision_year"]).size()
        total_cy = cy_counts.sum()
        cy_weights = cy_counts.apply(lambda c: total_cy / (cy_counts.shape[0] * c) if c > 0 else np.nan)
        country_year_weight = df.apply(
            lambda row: cy_weights.get((row["country_code_clean"], row["decision_year"]), np.nan), axis=1
        )
    else:
        country_year_weight = pd.Series(np.nan, index=df.index)
    return country_weight, country_year_weight


def build_design_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    df = pd.read_csv(WIDE_CSV)
    df["country_code_clean"] = df["country_code"].apply(harmonise_country)

    df["decision_date_parsed"] = pd.to_datetime(df["decision_date"], utc=True, errors="coerce")
    df["days_since_gdpr"] = (df["decision_date_parsed"] - GDPR_START).dt.days

    df["enforcement_severity_index"] = df["fine_positive"] + df["severity_measures_present"]

    df["time_observed"] = df["decision_year"].notna()

    # Derive inverse probability weights for time-observed subset
    features = pd.get_dummies(
        df[["country_code_clean", "breach_case", "fine_positive", "severity_measures_present"]]
        .fillna({"breach_case": 0, "fine_positive": 0, "severity_measures_present": 0}),
        columns=["country_code_clean"],
        drop_first=True,
    )
    logit = LogisticRegression(max_iter=1000)
    logit.fit(features, df["time_observed"].astype(int))
    prob = logit.predict_proba(features)[:, 1]
    prob = np.clip(prob, 0.05, 0.95)
    df["time_observed_prob"] = prob
    df.loc[df["time_observed"], "ipw_time"] = 1.0 / prob[df["time_observed"]]
    df.loc[~df["time_observed"], "ipw_time"] = np.nan

    country_weight, country_year_weight = compute_country_weights(df)
    df["country_weight"] = country_weight
    df["country_year_weight"] = country_year_weight

    # Parse breach type and sensitive data fields from raw answers
    breach_allowed = {
        "TECHNICAL_FAILURE",
        "ORGANIZATIONAL_FAILURE",
        "CYBER_ATTACK",
        "HUMAN_ERROR",
        "SYSTEM_MALFUNCTION",
    }
    sensitive_allowed = {
        "ARTICLE_9_SPECIAL_CATEGORY",
        "ARTICLE_10_CRIMINAL",
        "NEITHER",
    }

    breach_results = df["raw_q21"].apply(lambda val: parse_multiselect(val, breach_allowed))
    df["breach_types_status"] = breach_results.apply(lambda res: res.status)
    df["breach_types_conflict"] = breach_results.apply(lambda res: res.conflict)
    for token in sorted(breach_allowed):
        df[f"breach_type_{token.lower()}"] = breach_results.apply(
            lambda res: float(token in res.tokens) if res.status == "DISCUSSED" else np.nan
        )

    sensitive_results = df["raw_q25"].apply(lambda val: parse_multiselect(val, sensitive_allowed))
    df["special_data_status"] = sensitive_results.apply(lambda res: res.status)
    df["special_data_conflict"] = sensitive_results.apply(lambda res: res.conflict)
    df["special_data_article9"] = sensitive_results.apply(
        lambda res: float("ARTICLE_9_SPECIAL_CATEGORY" in res.tokens) if res.status == "DISCUSSED" else np.nan
    )
    df["special_data_article10"] = sensitive_results.apply(
        lambda res: float("ARTICLE_10_CRIMINAL" in res.tokens) if res.status == "DISCUSSED" else np.nan
    )
    df["special_data_neither"] = sensitive_results.apply(
        lambda res: float("NEITHER" in res.tokens) if res.status == "DISCUSSED" else np.nan
    )

    # Art. 33/34 enumerations
    df["art33_notification_required"] = df["raw_q17"].fillna("").replace("", np.nan)
    df["art33_submitted"] = df["raw_q18"].fillna("").replace("", np.nan)
    df["art33_submission_timing"] = df["raw_q19"].fillna("").replace("", np.nan)
    df["art33_delay_amount"] = df["raw_q20"].fillna("").replace("", np.nan)
    df["art34_notified"] = df["raw_q26"].fillna("").replace("", np.nan)
    df["art34_notification_required"] = df["raw_q27"].fillna("").replace("", np.nan)
    df["initiation_channel"] = df["raw_q15"].fillna("").replace("", np.nan)

    # Article 22 cause (internal/external)
    df["breach_cause"] = df["raw_q22"].fillna("").replace("", np.nan)

    # Vulnerable groups / remedial actions: drop exclusivity conflicts
    conflict_columns = [
        "q46_vuln_exclusivity_conflict",
        "q47_remedial_exclusivity_conflict",
        "q53_powers_exclusivity_conflict",
    ]
    conflict_mask = df[conflict_columns].fillna(0).sum(axis=1) == 0
    df = df.loc[conflict_mask].copy()

    df["vulnerable_status"] = df["q46_vuln_status"]
    vuln_cols = [
        "q46_vuln_CHILDREN",
        "q46_vuln_ELDERLY",
        "q46_vuln_PATIENTS",
        "q46_vuln_EMPLOYEES",
        "q46_vuln_FINANCIALLY_VULNERABLE",
    ]
    valid_vuln = df["vulnerable_status"].isin(["DISCUSSED", "NONE_MENTIONED"])
    df.loc[~valid_vuln, vuln_cols] = np.nan
    df["vulnerable_any"] = np.where(valid_vuln, df[vuln_cols].fillna(0).sum(axis=1) > 0, np.nan)

    df["remedial_status"] = df["q47_remedial_status"]
    remedial_cols = [
        "q47_remedial_IMMEDIATE_CESSATION",
        "q47_remedial_SYSTEM_UPGRADES",
        "q47_remedial_POLICY_CHANGES",
        "q47_remedial_STAFF_TRAINING",
        "q47_remedial_EXTERNAL_AUDIT",
        "q47_remedial_DPO_APPOINTMENT",
        "q47_remedial_COMPENSATION_TO_SUBJECTS",
    ]
    valid_rem = df["remedial_status"].isin(["DISCUSSED", "NONE_MENTIONED"])
    df.loc[~valid_rem, remedial_cols] = np.nan
    df["remedial_any"] = np.where(
        valid_rem,
        (df[remedial_cols].fillna(0).sum(axis=1) > 0).astype(float),
        np.nan,
    )

    # Prepare outputs
    keep_columns = [
        "decision_id",
        "country_code_clean",
        "country_group",
        "decision_date_parsed",
        "decision_year",
        "decision_quarter",
        "days_since_gdpr",
        "time_observed",
        "time_observed_prob",
        "ipw_time",
        "country_weight",
        "country_year_weight",
        "fine_eur",
        "fine_positive",
        "fine_log1p",
        "severity_measures_present",
        "enforcement_severity_index",
        "breach_case",
        "breach_types_status",
        "breach_types_conflict",
        "special_data_status",
        "special_data_conflict",
        "special_data_article9",
        "special_data_article10",
        "special_data_neither",
        "vulnerable_status",
        "vulnerable_any",
        "remedial_status",
        "remedial_any",
        "art33_notification_required",
        "art33_submitted",
        "art33_submission_timing",
        "art33_delay_amount",
        "art34_notified",
        "art34_notification_required",
        "initiation_channel",
        "breach_cause",
    ]
    keep_columns.extend([f"breach_type_{token.lower()}" for token in sorted(breach_allowed)])
    keep_columns.extend(vuln_cols)
    keep_columns.extend(remedial_cols)

    design = df[keep_columns].copy()

    design.to_parquet(ANALYSIS_DIR / "X_master.parquet", index=False)
    time_obs = design[df.loc[design.index, "time_observed"]].copy()
    time_obs.to_parquet(ANALYSIS_DIR / "X_timeobs.parquet", index=False)

    status_summary: Dict[str, Dict[str, int]] = {
        "breach_types_status": design["breach_types_status"].value_counts(dropna=False).to_dict(),
        "special_data_status": design["special_data_status"].value_counts(dropna=False).to_dict(),
        "vulnerable_status": design["vulnerable_status"].value_counts(dropna=False).to_dict(),
        "remedial_status": design["remedial_status"].value_counts(dropna=False).to_dict(),
    }

    brief_lines = [
        "# Stage 1 coverage summary",
        "",
        "## Status distributions",
    ]
    for field, counts in status_summary.items():
        brief_lines.append(f"### {field}")
        for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            brief_lines.append(f"- {key}: {value}")
        brief_lines.append("")
    brief_lines.append("## Country harmonisation")
    brief_lines.append("- Mapped 'COUNTRY OF THE DECIDING AUTHORITY: IRELAND (IE)' ? IE")
    brief_lines.append("- UK codes harmonised to GB")
    brief_lines.append("")
    brief_lines.append("## Weight notes")
    brief_lines.append(
        "- `country_weight` rescales each country to equal total weight across the sample"
    )
    brief_lines.append(
        "- `country_year_weight` equalises country-year cells where decision_year is observed"
    )

    (ANALYSIS_DIR / "stage1_data_brief.md").write_text("\n".join(brief_lines), encoding="utf-8")
    return design, time_obs, status_summary


def _prepare_features(df: pd.DataFrame, *, include_days: bool = True) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    raw_country = df["country_code_clean"].fillna("UNKNOWN")
    top_countries = {"ES", "IT", "PL", "GB", "FR"}
    features["country_code"] = raw_country.where(raw_country.isin(top_countries), "OTHER")
    # Aggregate initiation channel into 4 buckets
    channel_map = {
        "BREACH_NOTIFICATION": "BREACH",
        "COMPLAINT": "COMPLAINT",
        "EX_OFFICIO_DPA_INITIATIVE": "EX_OFFICIO",
    }
    features["init_channel"] = df["initiation_channel"].map(channel_map).fillna("OTHER")
    features["special_article9"] = df["special_data_article9"].fillna(0)
    features["special_article10"] = df["special_data_article10"].fillna(0)
    features["special_none_explicit"] = df["special_data_neither"].fillna(0)
    features["special_status_not_applicable"] = (df["special_data_status"] == "NOT_APPLICABLE").astype(int)
    features["special_status_not_mentioned"] = (
        df["special_data_status"].isin(["NOT_MENTIONED", "NOT_DISCUSSED", "MISSING"])
    ).astype(int)
    features["vulnerable_any"] = df["vulnerable_any"].fillna(0).astype(float)
    features["remedial_any"] = df["remedial_any"].fillna(0).astype(float)
    features["breach_type_cyber"] = df["breach_type_cyber_attack"].fillna(0)
    features["breach_type_human_error"] = df["breach_type_human_error"].fillna(0)
    features["breach_type_technical"] = df["breach_type_technical_failure"].fillna(0)
    features["breach_cause_external"] = (df["breach_cause"] == "EXTERNAL_CAUSE").astype(int)
    features["breach_cause_internal"] = (df["breach_cause"] == "INTERNAL_CAUSE").astype(int)
    if include_days:
        features["days_since_gdpr"] = df["days_since_gdpr"].fillna(0)
        features["days_since_gdpr_missing"] = df["days_since_gdpr"].isna().astype(int)
    features = pd.get_dummies(features, columns=["country_code", "init_channel"], drop_first=True)
    return features


def estimate_aipw(df: pd.DataFrame, treatment: pd.Series, outcome: pd.Series,
                  outcome_type: str) -> Tuple[float, Tuple[float, float]]:
    features = _prepare_features(df)
    feature_matrix = features.values
    treat = treatment.values.astype(int)

    prop_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    prop_model.fit(feature_matrix, treat)
    p_hat = prop_model.predict_proba(feature_matrix)[:, 1]
    p_hat = np.clip(p_hat, 0.05, 0.95)

    if outcome_type == "binary":
        y = outcome.values.astype(int)
        mu1_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        mu0_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        mu1_model.fit(feature_matrix[treat == 1], y[treat == 1])
        mu0_model.fit(feature_matrix[treat == 0], y[treat == 0])
        mu1 = mu1_model.predict_proba(feature_matrix)[:, 1]
        mu0 = mu0_model.predict_proba(feature_matrix)[:, 1]
    else:
        y = outcome.values.astype(float)
        mu1_model = LinearRegression()
        mu0_model = LinearRegression()
        mu1_model.fit(feature_matrix[treat == 1], y[treat == 1])
        mu0_model.fit(feature_matrix[treat == 0], y[treat == 0])
        mu1 = mu1_model.predict(feature_matrix)
        mu0 = mu0_model.predict(feature_matrix)

    aipw_scores = (mu1 - mu0
                   + treat * (y - mu1) / p_hat
                   - (1 - treat) * (y - mu0) / (1 - p_hat))
    ate = aipw_scores.mean()

    # Bootstrap for CI
    rng = np.random.default_rng(42)
    boot = []
    index = np.arange(len(df))
    for _ in range(BOOTSTRAP_REPS):
        sample_idx = rng.choice(index, size=len(index), replace=True)
        try:
            boot_df = df.iloc[sample_idx]
            boot_treat = treatment.iloc[sample_idx]
            boot_outcome = outcome.iloc[sample_idx]
            boot_features_df = _prepare_features(boot_df)
            boot_matrix = boot_features_df.values
            prop = LogisticRegression(max_iter=1000, class_weight="balanced")
            prop.fit(boot_matrix, boot_treat.values.astype(int))
            p_b = np.clip(prop.predict_proba(boot_matrix)[:, 1], 0.05, 0.95)
            if outcome_type == "binary":
                y_b = boot_outcome.values.astype(int)
                mu1_m = LogisticRegression(max_iter=1000, class_weight="balanced")
                mu0_m = LogisticRegression(max_iter=1000, class_weight="balanced")
                mu1_m.fit(boot_matrix[boot_treat == 1], y_b[boot_treat == 1])
                mu0_m.fit(boot_matrix[boot_treat == 0], y_b[boot_treat == 0])
                mu1_b = mu1_m.predict_proba(boot_matrix)[:, 1]
                mu0_b = mu0_m.predict_proba(boot_matrix)[:, 1]
            else:
                y_b = boot_outcome.values.astype(float)
                mu1_m = LinearRegression()
                mu0_m = LinearRegression()
                mu1_m.fit(boot_matrix[boot_treat == 1], y_b[boot_treat == 1])
                mu0_m.fit(boot_matrix[boot_treat == 0], y_b[boot_treat == 0])
                mu1_b = mu1_m.predict(boot_matrix)
                mu0_b = mu0_m.predict(boot_matrix)
            score = (mu1_b - mu0_b
                     + boot_treat.values * (y_b - mu1_b) / p_b
                     - (1 - boot_treat.values) * (y_b - mu0_b) / (1 - p_b))
            boot.append(score.mean())
        except Exception:
            continue
    if boot:
        lower, upper = np.percentile(boot, [2.5, 97.5])
    else:
        lower = upper = np.nan
    return ate, (lower, upper)


def run_stage2(design: pd.DataFrame) -> None:
    results = {}

    # 2a) 72-hour timing effect
    timing_mask = (
        (design["breach_case"] == 1)
        & (design["art33_notification_required"] == "YES_REQUIRED")
        & (design["art33_submission_timing"].isin(["YES_WITHIN_72H", "NO_LATE"]))
    )
    timing_df = design.loc[timing_mask].copy()
    timing_df["late_notification"] = (timing_df["art33_submission_timing"] == "NO_LATE").astype(int)
    if len(timing_df) >= 20 and timing_df["late_notification"].nunique() == 2:
        results["timing_fine_positive"] = estimate_aipw(
            timing_df,
            timing_df["late_notification"],
            timing_df["fine_positive"],
            outcome_type="binary",
        )
        results["timing_fine_log1p"] = estimate_aipw(
            timing_df,
            timing_df["late_notification"],
            timing_df["fine_log1p"],
            outcome_type="continuous",
        )
        results["timing_severity_index"] = estimate_aipw(
            timing_df,
            timing_df["late_notification"],
            timing_df["enforcement_severity_index"],
            outcome_type="continuous",
        )
    else:
        results["timing_fine_positive"] = (np.nan, (np.nan, np.nan))
        results["timing_fine_log1p"] = (np.nan, (np.nan, np.nan))
        results["timing_severity_index"] = (np.nan, (np.nan, np.nan))

    # 2b) Subject notification effect
    art34_mask = (
        (design["breach_case"] == 1)
        & (design["art34_notification_required"] == "YES_REQUIRED")
        & (design["art34_notified"].isin(["YES_NOTIFIED", "NO_NOT_NOTIFIED", "PARTIALLY_NOTIFIED", "NOTIFIED"]))
    )
    notify_df = design.loc[art34_mask].copy()
    notify_df["subjects_notified"] = notify_df["art34_notified"].isin(["YES_NOTIFIED", "NOTIFIED"]).astype(int)
    if len(notify_df) >= 20 and notify_df["subjects_notified"].nunique() == 2:
        results["notify_fine_positive"] = estimate_aipw(
            notify_df,
            notify_df["subjects_notified"],
            notify_df["fine_positive"],
            outcome_type="binary",
        )
        results["notify_fine_log1p"] = estimate_aipw(
            notify_df,
            notify_df["subjects_notified"],
            notify_df["fine_log1p"],
            outcome_type="continuous",
        )
        results["notify_severity_index"] = estimate_aipw(
            notify_df,
            notify_df["subjects_notified"],
            notify_df["enforcement_severity_index"],
            outcome_type="continuous",
        )
    else:
        results["notify_fine_positive"] = (np.nan, (np.nan, np.nan))
        results["notify_fine_log1p"] = (np.nan, (np.nan, np.nan))
        results["notify_severity_index"] = (np.nan, (np.nan, np.nan))

    # Persist results
    result_rows = []
    for key, (estimate, ci) in results.items():
        result_rows.append({
            "analysis": key,
            "estimate": estimate,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
        })
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(ANALYSIS_DIR / "stage2_results.csv", index=False)

    summary_lines = [
        "# Stage 2 causal estimates",
        "",
    ]
    for _, row in results_df.iterrows():
        summary_lines.append(
            f"- {row['analysis']}: estimate={row['estimate']:.4f} (95% CI {row['ci_lower']:.4f}, {row['ci_upper']:.4f})"
        )
    (ANALYSIS_DIR / "stage2_results.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    design, _, _ = build_design_matrices()
    run_stage2(design)


if __name__ == "__main__":
    main()
