"""GDPR evenness analysis toolkit."""
from .config import (
    EvennessPaths,
    FACTS_CONFIG,
    DEFAULT_MATCHING_WITHIN,
    DEFAULT_MATCHING_CROSS,
    LENIENCY_RANDOM_SLOPE_DRIVERS,
    ROBUSTNESS_SCENARIOS,
    required_fact_columns,
)
from .data import build_fact_matrix
from .matching import perform_matching
from .modeling import fit_logistic, fit_ols, fit_mixed_effects, model_to_dict
from .leniency import compute_leniency_index
from .variance import intraclass_correlation, variance_summary
from .decomposition import run_oaxaca_blinder
from .robustness import run_robustness_suite
from .predictive import gradient_boosting_diagnostics

__all__ = [
    "EvennessPaths",
    "FACTS_CONFIG",
    "DEFAULT_MATCHING_WITHIN",
    "DEFAULT_MATCHING_CROSS",
    "LENIENCY_RANDOM_SLOPE_DRIVERS",
    "ROBUSTNESS_SCENARIOS",
    "required_fact_columns",
    "build_fact_matrix",
    "perform_matching",
    "fit_logistic",
    "fit_ols",
    "fit_mixed_effects",
    "model_to_dict",
    "compute_leniency_index",
    "intraclass_correlation",
    "variance_summary",
    "run_oaxaca_blinder",
    "run_robustness_suite",
    "gradient_boosting_diagnostics",
]
