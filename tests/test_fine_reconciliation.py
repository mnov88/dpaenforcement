from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from scripts.analysis.fine_reconciliation import (
    ComparisonSummary,
    FineRecord,
    apply_human_overrides,
    compare_fines,
    load_human_fines,
    summarise_comparisons,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_human_fines_parses_currency_and_amount(tmp_path: Path) -> None:
    human_path = tmp_path / "human.csv"
    fieldnames = ["\ufeffID", "Fine", "Fine_Numeric"]
    rows = [
        {"\ufeffID": "Case1", "Fine": "5,000 EUR", "Fine_Numeric": "5000"},
        {"\ufeffID": "Case2", "Fine": "10 000 GBP", "Fine_Numeric": "10000"},
        {"\ufeffID": "Case3", "Fine": "150000", "Fine_Numeric": ""},
    ]
    _write_csv(human_path, fieldnames, rows)

    fines = load_human_fines(human_path)

    assert "Case1" in fines
    assert fines["Case1"].currency == "EUR"
    assert fines["Case1"].amount_native == pytest.approx(5000.0)
    assert fines["Case1"].amount_eur == pytest.approx(5000.0)

    assert fines["Case2"].currency == "GBP"
    assert fines["Case2"].amount_native == pytest.approx(10000.0)
    assert fines["Case2"].amount_eur == pytest.approx(11700.0)

    # Without an explicit currency we assume the amount is already in EUR.
    assert fines["Case3"].currency == "EUR"
    assert fines["Case3"].amount_eur == pytest.approx(150000.0)


def test_apply_human_overrides_updates_numeric_fields(tmp_path: Path) -> None:
    ai_rows = [
        {
            "decision_id": "Case1",
            "fine_eur": "100.000000",
            "fine_numeric_valid": "1",
            "fine_positive": "1",
            "fine_log1p": f"{math.log1p(100):.6f}",
            "fine_outlier_flag": "0",
            "fine_raw": "TYPE:NUMBER 100",
            "fine_status": "DISCUSSED",
            "fine_error": "",
            "turnover_eur": "1000.000000",
            "fine_to_turnover_ratio": f"{100/1000:.8f}",
        },
        {
            "decision_id": "Case2",
            "fine_eur": "200.000000",
            "fine_numeric_valid": "1",
            "fine_positive": "1",
            "fine_log1p": f"{math.log1p(200):.6f}",
            "fine_outlier_flag": "0",
            "fine_raw": "TYPE:NUMBER 200",
            "fine_status": "DISCUSSED",
            "fine_error": "",
            "turnover_eur": "600.000000",
            "fine_to_turnover_ratio": f"{200/600:.8f}",
        },
        {
            "decision_id": "Case3",
            "fine_eur": "",
            "fine_numeric_valid": "0",
            "fine_positive": "0",
            "fine_log1p": "",
            "fine_outlier_flag": "0",
            "fine_raw": "",
            "fine_status": "NOT_MENTIONED",
            "fine_error": "",
            "turnover_eur": "",
            "fine_to_turnover_ratio": "",
        },
        {
            "decision_id": "Case4",
            "fine_eur": "50.000000",
            "fine_numeric_valid": "1",
            "fine_positive": "1",
            "fine_log1p": f"{math.log1p(50):.6f}",
            "fine_outlier_flag": "0",
            "fine_raw": "TYPE:NUMBER 50",
            "fine_status": "DISCUSSED",
            "fine_error": "",
            "turnover_eur": "",
            "fine_to_turnover_ratio": "",
        },
    ]

    human_path = tmp_path / "human.csv"
    fieldnames = ["\ufeffID", "Fine", "Fine_Numeric"]
    human_rows = [
        {"\ufeffID": "Case1", "Fine": "100 EUR", "Fine_Numeric": "100"},
        {"\ufeffID": "Case2", "Fine": "300 EUR", "Fine_Numeric": "300"},
        {"\ufeffID": "Case3", "Fine": "1,000 GBP", "Fine_Numeric": "1000"},
    ]
    _write_csv(human_path, fieldnames, human_rows)
    human_fines = load_human_fines(human_path)

    updated_rows, overrides = apply_human_overrides(ai_rows, human_fines)

    assert overrides == ["Case1", "Case2", "Case3"]

    # Case1 remains unchanged but receives human bookkeeping columns.
    row_case1 = next(row for row in updated_rows if row["decision_id"] == "Case1")
    assert row_case1["fine_eur"] == "100.000000"
    assert row_case1["fine_amount_human_native"] == "100.000000"
    assert row_case1["fine_currency_human"] == "EUR"

    # Case2 should now reflect the human amount and update the ratio.
    row_case2 = next(row for row in updated_rows if row["decision_id"] == "Case2")
    assert row_case2["fine_eur"] == "300.000000"
    assert float(row_case2["fine_to_turnover_ratio"]) == pytest.approx(
        0.5, rel=1e-6
    )
    assert float(row_case2["fine_log1p"]) == pytest.approx(
        math.log1p(300), rel=1e-6
    )

    # Case3 receives a converted GBP amount.
    row_case3 = next(row for row in updated_rows if row["decision_id"] == "Case3")
    assert row_case3["fine_eur"] == "1170.000000"
    assert row_case3["fine_numeric_valid"] == "1"
    assert row_case3["fine_positive"] == "1"
    assert row_case3["fine_status"] == "DISCUSSED"
    assert row_case3["fine_amount_human_native"] == "1000.000000"
    assert row_case3["fine_currency_human"] == "GBP"

    # Case4 has no human override.
    row_case4 = next(row for row in updated_rows if row["decision_id"] == "Case4")
    assert row_case4["fine_eur"] == "50.000000"
    assert "fine_amount_human_native" not in row_case4


def test_compare_fines_classifies_conflicts() -> None:
    ai_fines = {
        "Case1": FineRecord("Case1", 100.0, 100.0, "EUR", "100", "ai"),
        "Case2": FineRecord("Case2", 200.0, 200.0, "EUR", "200", "ai"),
        "Case3": FineRecord("Case3", None, None, None, "", "ai"),
    }
    human_fines = {
        "Case1": FineRecord("Case1", 100.5, 100.5, "EUR", "100.5 EUR", "human"),
        "Case2": FineRecord("Case2", 350.0, 1500.0, "PLN", "1500 PLN", "human"),
        "Case4": FineRecord("Case4", 50.0, 50.0, "EUR", "50", "human"),
    }

    comparisons = compare_fines(ai_fines, human_fines, tolerance=1.0)

    statuses = {comp.decision_id: comp.status for comp in comparisons}
    assert statuses["Case1"] == "MATCH"
    assert statuses["Case2"] == "CONFLICT"
    assert statuses["Case3"] == "MISSING_BOTH"
    assert statuses["Case4"] == "AI_MISSING"

    summary = summarise_comparisons(comparisons)
    assert isinstance(summary, ComparisonSummary)
    assert summary.total == 4
    assert summary.matches == 1
    assert summary.conflicts == 1
    assert summary.ai_missing == 1
    assert summary.human_missing == 0
    assert summary.missing_both == 1
