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
    match_within_csv: Path = Path("outputs/evenness/matches_within.csv")
    match_cross_csv: Path = Path("outputs/evenness/matches_cross.csv")
    model_dir: Path = Path("outputs/evenness/models")
    leniency_csv: Path = Path("outputs/evenness/leniency_index.csv")
    leniency_plot: Path = Path("outputs/evenness/leniency_map.png")
    variance_csv: Path = Path("outputs/evenness/variance_components.csv")
    decomposition_dir: Path = Path("outputs/evenness/decomposition")
    robustness_dir: Path = Path("outputs/evenness/robustness")

    def ensure(self) -> None:
        """Create parent directories for all registered artefacts."""

        for path in (
            self.feature_cache,
            self.match_within_csv,
            self.match_cross_csv,
            self.model_dir,
            self.leniency_csv,
            self.leniency_plot,
            self.variance_csv,
            self.decomposition_dir,
            self.robustness_dir,
        ):
            parent = Path(path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)


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
