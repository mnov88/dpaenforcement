import pandas as pd

from scripts.evenness import cli


def test_build_formula_includes_fixed_effects():
    df = pd.DataFrame(
        {
            "decision_id": ["A", "B"],
            "breach_case": ["YES", "NO"],
            "organization_size_tier": ["SME", "SME"],
            "organization_type": ["ORGANIZATION", "ORGANIZATION"],
            "case_origin": ["COMPLAINT", "BREACH_NOTIFICATION"],
            "country_code": ["FR", "DE"],
            "dpa_name_canonical": ["CNIL", "BfDI"],
            "n_principles_violated": [1, 0],
            "n_corrective_measures": [2, 1],
            "days_since_gdpr": [100, 200],
            "q21_breach_types_TECHNICAL_FAILURE": [1, 0],
            "q21_breach_types_status": ["DISCUSSED", "DISCUSSED"],
            "q21_breach_types_coverage_status": ["DISCUSSED", "DISCUSSED"],
        }
    )
    formula = cli._build_formula("fine_positive", df)
    assert formula.startswith("fine_positive ~")
    assert "C(country_code)" in formula
    assert "q21_breach_types_TECHNICAL_FAILURE" in formula


def test_indicator_columns_ignore_guardrails():
    df = pd.DataFrame(
        {
            "decision_id": ["1"],
            "q46_vuln_status": ["DISCUSSED"],
            "q46_vuln_coverage_status": ["DISCUSSED"],
            "q46_vuln_CHILDREN": [1],
            "q46_vuln_NONE_MENTIONED": [0],
        }
    )
    cols = cli._indicator_columns(df)
    assert "q46_vuln_CHILDREN" in cols
    assert "q46_vuln_status" not in cols
