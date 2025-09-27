import pandas as pd

from scripts.evenness import (
    DEFAULT_MATCHING_CROSS,
    DEFAULT_MATCHING_WITHIN,
    FACTS_CONFIG,
    required_fact_columns,
)


def test_matching_specs_have_unique_features():
    assert len(set(DEFAULT_MATCHING_WITHIN.all_features())) == len(DEFAULT_MATCHING_WITHIN.all_features())
    assert len(set(DEFAULT_MATCHING_CROSS.all_features())) == len(DEFAULT_MATCHING_CROSS.all_features())


def test_required_fact_columns_cover_outcomes():
    cols = required_fact_columns()
    for field in FACTS_CONFIG.outcome_columns:
        assert field in cols


def test_multi_value_prefixes_produce_guardrails():
    df = pd.DataFrame(
        {
            "q21_breach_types_coverage_status": ["DISCUSSED"],
            "q21_breach_types_status": ["DISCUSSED"],
            "q21_breach_types_TECHNICAL_FAILURE": [1],
        }
    )
    assert any(col.startswith("q21_breach_types_") for col in df.columns)
