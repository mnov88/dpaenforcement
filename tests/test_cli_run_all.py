import json
from pathlib import Path

import pytest

from scripts import cli


@pytest.fixture
def tmp_files(tmp_path: Path) -> dict[str, Path]:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Sample prompt", encoding="utf-8")
    input_csv = tmp_path / "raw.csv"
    input_csv.write_text("ID,response\nX1,Answer 1: TEST\n", encoding="utf-8")
    return {
        "prompt": prompt,
        "input": input_csv,
        "enum": tmp_path / "enum.json",
        "wide": tmp_path / "wide.csv",
        "validation": tmp_path / "validation.json",
        "long_dir": tmp_path / "long",
        "consistency": tmp_path / "consistency.json",
        "qa": tmp_path / "qa.csv",
        "feature_parquet": tmp_path / "fm.parquet",
        "feature_meta": tmp_path / "fm.json",
        "evenness_wide": tmp_path / "evenness.csv",
    }


def test_run_all_triggers_optional_analysis(monkeypatch, tmp_files):
    calls: list[str] = []

    monkeypatch.setattr(cli, "build_enum_whitelist", lambda *args, **kwargs: {"enums": []})

    def fake_clean(input_csv, out_csv, validation_json):
        out_csv.write_text("decision_id\nX1\n", encoding="utf-8")
        validation_json.write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(cli, "clean_csv_to_wide", fake_clean)

    class DummyEmitter:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def emit_from_csv(self, *args, **kwargs):
            calls.append("long")

    monkeypatch.setattr(cli, "LongEmitter", DummyEmitter)
    monkeypatch.setattr(cli, "run_consistency_checks", lambda *a, **k: calls.append("consistency"))
    monkeypatch.setattr(cli, "create_qa_summary", lambda *a, **k: calls.append("qa"))
    monkeypatch.setattr(cli, "_run_feature_matrix", lambda *a, **k: calls.append("feature"))
    monkeypatch.setattr(cli, "_run_evenness_pipeline", lambda *a, **k: calls.append("evenness"))

    argv = [
        "run-all",
        "--prompt-path",
        str(tmp_files["prompt"]),
        "--enum-out",
        str(tmp_files["enum"]),
        "--input-csv",
        str(tmp_files["input"]),
        "--out-csv",
        str(tmp_files["wide"]),
        "--validation-json",
        str(tmp_files["validation"]),
        "--long-dir",
        str(tmp_files["long_dir"]),
        "--consistency-json",
        str(tmp_files["consistency"]),
        "--qa-summary-csv",
        str(tmp_files["qa"]),
        "--skip-fine-reconciliation",
        "--build-feature-matrix",
        "--feature-matrix-parquet",
        str(tmp_files["feature_parquet"]),
        "--feature-matrix-metadata",
        str(tmp_files["feature_meta"]),
        "--run-evenness",
        "--evenness-light",
        "--evenness-wide-csv",
        str(tmp_files["evenness_wide"]),
        "--evenness-phase-three-outcome",
        "fine_log1p",
        "--evenness-phase-three-permutations",
        "5",
    ]

    parser = cli.build_parser()
    args = parser.parse_args(argv)
    cli.cmd_run_all(args)

    assert "feature" in calls
    assert "evenness" in calls
    assert "long" in calls
    assert "consistency" in calls
    assert "qa" in calls
