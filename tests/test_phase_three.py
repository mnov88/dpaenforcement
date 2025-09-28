import pandas as pd

from scripts.evenness.interaction import interaction_scan
from scripts.evenness.phase_three import estimate_notification_effect, estimate_timing_effect


def _demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "art33_required": ["YES"] * 8,
            "art33_delay_amount": [-10, -6, -2, -1, 5, 12, 24, 36],
            "art33_submission_timing": [
                "PROMPT",
                "PROMPT",
                "PROMPT",
                "PROMPT",
                "LATE",
                "LATE",
                "LATE",
                "LATE",
            ],
            "fine_positive": [0, 0, 0, 1, 1, 1, 1, 1],
            "fine_log1p": [0.0, 0.1, 0.2, 0.3, 0.8, 1.2, 1.4, 1.6],
        }
    )


def test_estimate_timing_effect_returns_results():
    df = _demo_dataframe()
    effects, placebos = estimate_timing_effect(df, ["fine_positive", "fine_log1p"], bandwidth=48, donut=0)
    assert not effects.empty
    assert (effects["lever"] == "art33_timing").all()
    assert set(effects["result_type"].unique()).issubset({"reduced_form", "local_ate"})
    # Placebo table may be empty if offsets trim all rows, but function should return a DataFrame
    assert placebos is not None


def test_estimate_notification_effect_reports_ate_and_att():
    df = pd.DataFrame(
        {
            "art34_required": ["YES"] * 6,
            "subjects_notified": ["YES", "NO", "YES", "NO", "YES", "NO"],
            "fine_log1p": [1.2, 0.5, 1.4, 0.6, 1.5, 0.7],
            "n_principles_violated": [1, 1, 2, 2, 3, 1],
            "n_corrective_measures": [0, 1, 1, 0, 1, 0],
            "days_since_gdpr": [100, 200, 300, 400, 500, 250],
            "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY": [0, 1, 0, 1, 0, 0],
        }
    )
    features = [
        "n_principles_violated",
        "n_corrective_measures",
        "days_since_gdpr",
        "q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY",
    ]
    results = estimate_notification_effect(df, ["fine_log1p"], features)
    assert {"ate", "att"}.issubset(set(results["result_type"]))
    assert (results["lever"] == "subjects_notified").all()


def test_interaction_scan_supports_custom_group_field():
    df = pd.DataFrame(
        {
            "fine_log1p": [1.0, 1.1, 0.9, 1.4],
            "driver": [0, 1, 0, 1],
            "country_code": ["A", "A", "B", "B"],
            "dpa_name_canonical": ["DA", "DA", "DB", "DB"],
        }
    )
    base_formula = "fine_log1p ~ driver"
    result = interaction_scan(df, "fine_log1p", base_formula, ["driver"], group_field="dpa_name_canonical")
    assert "pvalue" in result.columns
    assert (result["group_field"] == "dpa_name_canonical").all()
