"""Configuration objects and defaults for the GDPR evenness analysis toolkit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvennessPaths:
    """Centralized file locations used by the CLI."""

    wide_csv: Path = Path("outputs/cleaned_wide_latest.csv")
    feature_cache: Path = Path("outputs/evenness/fact_matrix.parquet")
    omniscan_dir: Path = Path("outputs/evenness/omniscan")
    feature_universe_json: Path = Path("outputs/evenness/omniscan/features_universe.json")
    coverage_ledger_csv: Path = Path("outputs/evenness/omniscan/coverage_ledger.csv")
    coverage_checklist_csv: Path = Path("outputs/evenness/omniscan/no_feature_left_behind.csv")
    importance_heatmap_csv: Path = Path("outputs/evenness/omniscan/importance_heatmap.csv")
    interaction_map_csv: Path = Path("outputs/evenness/omniscan/interaction_map.csv")
    block_importance_csv: Path = Path("outputs/evenness/omniscan/block_importance.csv")
    shap_country_csv: Path = Path("outputs/evenness/omniscan/shap_country_summary.csv")
    shap_dpa_csv: Path = Path("outputs/evenness/omniscan/shap_dpa_summary.csv")
    sage_importance_csv: Path = Path("outputs/evenness/omniscan/sage_importance.csv")
    specification_curve_csv: Path = Path("outputs/evenness/omniscan/specification_curve.csv")
    stability_selection_csv: Path = Path("outputs/evenness/omniscan/stability_selection.csv")
    knockoff_results_csv: Path = Path("outputs/evenness/omniscan/knockoff_results.csv")
    robust_driver_csv: Path = Path("outputs/evenness/omniscan/robust_driver_list.csv")
    crt_results_csv: Path = Path("outputs/evenness/omniscan/crt_results.csv")
    jurisdiction_effects_csv: Path = Path("outputs/evenness/omniscan/jurisdiction_effects.csv")
    heterogeneity_csv: Path = Path("outputs/evenness/omniscan/heterogeneity_map.csv")
    network_edges_csv: Path = Path("outputs/evenness/omniscan/network_edges.csv")
    community_summary_csv: Path = Path("outputs/evenness/omniscan/network_communities.csv")
    risk_band_parity_csv: Path = Path("outputs/evenness/omniscan/risk_band_parity.csv")
    distribution_contrasts_csv: Path = Path("outputs/evenness/omniscan/risk_band_distribution.csv")
    x_full: Path = Path("outputs/evenness/X_full.parquet")
    x_timeobs: Path = Path("outputs/evenness/X_timeobs.parquet")
    match_within_csv: Path = Path("outputs/evenness/matches_within.csv")
    match_cross_csv: Path = Path("outputs/evenness/matches_cross.csv")
    twins_cem: Path = Path("outputs/evenness/twins_cem.parquet")
    twins_gower_within: Path = Path("outputs/evenness/twins_gower_within.parquet")
    twins_gower_cross: Path = Path("outputs/evenness/twins_gower_cross.parquet")
    twins_riskbands: Path = Path("outputs/evenness/twins_riskbands.parquet")
    model_dir: Path = Path("outputs/evenness/models")
    leniency_csv: Path = Path("outputs/evenness/leniency_index.csv")
    leniency_plot: Path = Path("outputs/evenness/leniency_map.png")
    variance_csv: Path = Path("outputs/evenness/variance_components.csv")
    decomposition_dir: Path = Path("outputs/evenness/decomposition")
    robustness_dir: Path = Path("outputs/evenness/robustness")
    balance_csv: Path = Path("outputs/evenness/twin_balance_diagnostics.csv")
    coverage_dir: Path = Path("outputs/evenness/coverage")
    support_dir: Path = Path("outputs/evenness/support")
    harmonization_log: Path = Path("outputs/evenness/country_harmonization_log.csv")
    uniformity_dir: Path = Path("outputs/evenness/uniformity")
    uniformity_residuals: Path = Path("outputs/evenness/uniformity/residuals.parquet")
    uniformity_effects_csv: Path = Path("outputs/evenness/uniformity/jurisdiction_effects.csv")
    uniformity_joint_csv: Path = Path("outputs/evenness/uniformity/joint_tests.csv")
    uniformity_pairs_csv: Path = Path("outputs/evenness/uniformity/paired_tests.csv")
    uniformity_distribution_csv: Path = Path("outputs/evenness/uniformity/distribution_tests.csv")
    uniformity_quantiles_csv: Path = Path("outputs/evenness/uniformity/quantile_contrasts.csv")
    uniformity_calibration_csv: Path = Path("outputs/evenness/uniformity/calibration.csv")
    uniformity_variance_csv: Path = Path("outputs/evenness/uniformity/variance_components.csv")
    phase3_dir: Path = Path("outputs/evenness/phase_three")
    driver_leaderboard_csv: Path = Path("outputs/evenness/phase_three/driver_leaderboard.csv")
    interaction_country_csv: Path = Path("outputs/evenness/phase_three/country_interactions.csv")
    interaction_dpa_csv: Path = Path("outputs/evenness/phase_three/dpa_interactions.csv")
    decomposition_summary_csv: Path = Path("outputs/evenness/phase_three/decomposition_summary.csv")
    policy_dir: Path = Path("outputs/evenness/phase_three/policy")
    policy_estimates_csv: Path = Path("outputs/evenness/phase_three/policy/lever_estimates.csv")
    policy_placebo_csv: Path = Path("outputs/evenness/phase_three/policy/rd_placebos.csv")
    policy_plot: Path = Path("outputs/evenness/phase_three/policy/lever_effects.png")
    robustness_summary_csv: Path = Path("outputs/evenness/phase_three/robustness_summary.csv")
    shap_summary_csv: Path = Path("outputs/evenness/phase_three/shap_attributions.csv")
    randomization_csv: Path = Path("outputs/evenness/phase_three/randomization_inference.csv")
    insights_report: Path = Path("outputs/evenness/phase_three/insights_report.md")
    playbook_report: Path = Path("outputs/evenness/phase_three/playbook.md")
    environment_snapshot: Path = Path("outputs/evenness/phase_three/environment.txt")

    def ensure(self) -> None:
        """Create parent directories for all registered artefacts."""

        for path in (
            self.feature_cache,
            self.feature_universe_json,
            self.coverage_ledger_csv,
            self.coverage_checklist_csv,
            self.importance_heatmap_csv,
            self.interaction_map_csv,
            self.block_importance_csv,
            self.shap_country_csv,
            self.shap_dpa_csv,
            self.sage_importance_csv,
            self.specification_curve_csv,
            self.stability_selection_csv,
            self.knockoff_results_csv,
            self.robust_driver_csv,
            self.crt_results_csv,
            self.jurisdiction_effects_csv,
            self.heterogeneity_csv,
            self.network_edges_csv,
            self.community_summary_csv,
            self.risk_band_parity_csv,
            self.distribution_contrasts_csv,
            self.x_full,
            self.x_timeobs,
            self.match_within_csv,
            self.match_cross_csv,
            self.twins_cem,
            self.twins_gower_within,
            self.twins_gower_cross,
            self.twins_riskbands,
            self.model_dir,
            self.leniency_csv,
            self.leniency_plot,
            self.variance_csv,
            self.decomposition_dir,
            self.robustness_dir,
            self.balance_csv,
            self.coverage_dir,
            self.support_dir,
            self.harmonization_log,
            self.uniformity_residuals,
            self.uniformity_effects_csv,
            self.uniformity_joint_csv,
            self.uniformity_pairs_csv,
            self.uniformity_distribution_csv,
            self.uniformity_quantiles_csv,
            self.uniformity_calibration_csv,
            self.uniformity_variance_csv,
            self.driver_leaderboard_csv,
            self.interaction_country_csv,
            self.interaction_dpa_csv,
            self.decomposition_summary_csv,
            self.policy_estimates_csv,
            self.policy_placebo_csv,
            self.policy_plot,
            self.robustness_summary_csv,
            self.shap_summary_csv,
            self.randomization_csv,
            self.insights_report,
            self.playbook_report,
            self.environment_snapshot,
        ):
            parent = Path(path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.coverage_dir,
            self.support_dir,
            self.uniformity_dir,
            self.phase3_dir,
            self.policy_dir,
            self.omniscan_dir,
        ):
            Path(directory).expanduser().resolve().mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MatchingSpec:
    """Specification for hybrid exact/Gower nearest-neighbour matching."""

    exact_features: Sequence[str]
    gower_numeric: Sequence[str]
    gower_categorical: Sequence[str]
    caliper: float = 0.35
    neighbours: int = 3
    min_group_size: int = 2

    def all_features(self) -> Sequence[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for collection in (self.exact_features, self.gower_numeric, self.gower_categorical):
            for name in collection:
                if name not in seen:
                    ordered.append(name)
                    seen.add(name)
        return tuple(ordered)


@dataclass(frozen=True)
class FactsConfig:
    """Defines which columns comprise the 'facts-only' feature matrix."""

    single_value_columns: Sequence[str] = (
        "decision_id",
        "breach_case",
        "decision_year",
        "decision_quarter",
        "country_code",
        "dpa_name_canonical",
        "isic_section",
        "isic_code",
        "isic_desc",
        "organization_size_tier",
        "organization_type",
        "case_origin",
        "days_since_gdpr",
    )
    multi_value_prefixes: Sequence[str] = (
        "q21_breach_types",
        "q25_sensitive_data",
        "q46_vuln",
        "q47_remedial",
        "q53_powers",
    )
    binary_columns: Sequence[str] = (
        "fine_positive",
    )
    outcome_columns: Sequence[str] = (
        "fine_positive",
        "fine_eur",
        "fine_log1p",
        "enforcement_severity_index",
    )

    @property
    def guardrail_prefixes(self) -> Sequence[str]:
        """Prefixes that carry *_status/*_coverage metadata."""

        return self.multi_value_prefixes + (
            "q10_org_class",
        )


DEFAULT_MATCHING_WITHIN = MatchingSpec(
    exact_features=(
        "breach_case",
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
        "q25_sensitive_data_ARTICLE_10_CRIMINAL",
        "q25_sensitive_data_NEITHER",
        "q46_vuln_CHILDREN",
        "decision_year_bucket",
        "isic_section",
    ),
    gower_numeric=(
        "days_since_gdpr",
        "n_principles_violated",
        "n_corrective_measures",
    ),
    gower_categorical=(
        "country_code",
        "dpa_name_canonical",
        "organization_size_tier",
        "organization_type",
    ),
    caliper=0.3,
    neighbours=3,
)

DEFAULT_MATCHING_CROSS = MatchingSpec(
    exact_features=(
        "breach_case",
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
        "q25_sensitive_data_ARTICLE_10_CRIMINAL",
        "q25_sensitive_data_NEITHER",
        "q46_vuln_CHILDREN",
        "decision_year_bucket",
        "isic_section",
        "organization_type",
    ),
    gower_numeric=(
        "days_since_gdpr",
        "n_principles_violated",
        "n_corrective_measures",
    ),
    gower_categorical=(
        "organization_size_tier",
        "case_origin",
        "dpa_name_canonical",
    ),
    caliper=0.25,
    neighbours=5,
)


FACTS_CONFIG = FactsConfig()


LENIENCY_RANDOM_SLOPE_DRIVERS: Sequence[str] = (
    "breach_case",
    "q46_vuln_CHILDREN",
    "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
    "organization_size_tier",
)


ROBUSTNESS_SCENARIOS: Mapping[str, Mapping[str, object]] = {
    "reweight_country_year": {"weighting": "country_year"},
    "heckman_turnover": {"selection": "turnover"},
    "discussed_only": {"discussed_only": True},
    "winsorize_fines": {"winsorize": 0.99},
    "quantile_75": {"quantile": 0.75},
    "quantile_90": {"quantile": 0.90},
}


def required_fact_columns() -> set[str]:
    """Return the minimal set of columns required downstream."""

    cols: set[str] = set()
    cols.update(FACTS_CONFIG.single_value_columns)
    cols.update(FACTS_CONFIG.binary_columns)
    cols.update(FACTS_CONFIG.outcome_columns)
    for prefix in FACTS_CONFIG.multi_value_prefixes:
        cols.add(f"{prefix}_status")
        cols.add(f"{prefix}_coverage_status")
        cols.add(f"{prefix}_exclusivity_conflict")
    # multi-value indicator columns will be collected dynamically
    cols.update(
        {
            "n_principles_discussed",
            "n_principles_violated",
            "n_corrective_measures",
            "severity_measures_present",
            "remedy_only_case",
        }
    )
    return cols


__all__ = [
    "EvennessPaths",
    "MatchingSpec",
    "FactsConfig",
    "FACTS_CONFIG",
    "DEFAULT_MATCHING_WITHIN",
    "DEFAULT_MATCHING_CROSS",
    "LENIENCY_RANDOM_SLOPE_DRIVERS",
    "ROBUSTNESS_SCENARIOS",
    "required_fact_columns",
]
