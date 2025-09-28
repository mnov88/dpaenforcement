import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from scripts.evenness import omniscan


def _sample_wide_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_id": ["1", "2", "3"],
            "country_code": ["uk", "ES", "DE"],
            "dpa_name_canonical": ["ICO", "AEPD", "BfDI"],
            "fine_positive": [1, 0, 1],
            "fine_eur": [1000.0, 0.0, 500.0],
            "fine_log1p": [np.log1p(1000.0), 0.0, np.log1p(500.0)],
            "enforcement_severity_index": [0.5, 0.1, 0.3],
            "q21_breach_types_STATUS": [np.nan, np.nan, np.nan],
            "q21_breach_types_status": ["DISCUSSED", "NOT_MENTIONED", "DISCUSSED"],
            "q21_breach_types_coverage_status": ["DISCUSSED", "NOT_MENTIONED", "DISCUSSED"],
            "q21_breach_types_exclusivity_conflict": [0, 0, 0],
            "q21_breach_types_DATA_LOSS": [1, 0, 0],
            "q21_breach_types_UNAUTHORISED_ACCESS": [0, 0, 1],
            "art33_required": ["YES", "YES", "NO"],
            "art33_required_status": ["DISCUSSED", "DISCUSSED", "DISCUSSED"],
            "art33_submitted": ["YES", "NO", "NO"],
            "art33_submitted_status": ["DISCUSSED", "DISCUSSED", "DISCUSSED"],
            "art33_submission_timing": ["WITHIN_72H", "AFTER", "WITHIN_72H"],
            "art33_submission_timing_status": ["DISCUSSED", "DISCUSSED", "DISCUSSED"],
            "art34_required": ["YES", "NO", "YES"],
            "art34_required_status": ["DISCUSSED", "DISCUSSED", "DISCUSSED"],
            "subjects_notified": ["YES", "NO", "YES"],
            "subjects_notified_status": ["DISCUSSED", "DISCUSSED", "DISCUSSED"],
            "cross_border": ["YES", "NO", "NO"],
            "cross_border_status": ["DISCUSSED", "DISCUSSED", "DISCUSSED"],
            "n_principles_violated": [1, 0, 2],
            "n_corrective_measures": [2, 0, 3],
        }
    )


def test_feature_matrix_masks_non_discussed_statuses():
    df = _sample_wide_df()
    matrix, metadata, coverage, checklist = omniscan._build_feature_matrix(df)

    # Verify decision IDs preserved and country harmonisation applied via dummy expansion
    assert "decision_id" in matrix.columns
    assert any(meta.feature.startswith("country_code_") for meta in metadata)

    # Row with NOT_MENTIONED status should have NaN for underlying indicators
    masked_value = matrix.loc[matrix["decision_id"] == "2", "q21_breach_types_DATA_LOSS"].iloc[0]
    assert np.isnan(masked_value)

    # Coverage ledger captures status mix information
    record = coverage.loc[coverage["feature"] == "q21_breach_types_DATA_LOSS"].iloc[0]
    assert record["observed_share"] < 1.0

    # Checklist marks status column as covered
    covered = checklist.loc[checklist["column"] == "q21_breach_types_status", "covered"].iloc[0]
    assert bool(covered)


def test_risk_band_assignments_returns_summary():
    predictions = pd.Series([0.1, 0.9, 0.4, 0.6])
    outcome = pd.Series([0, 1, 0, 1])
    jurisdiction = pd.Series(["DE", "DE", "FR", "FR"])
    summary, assignments = omniscan._risk_band_assignments(predictions, outcome, jurisdiction, bands=2)

    assert set(summary["band"]) <= {0, 1}
    assert "band" in assignments.columns
