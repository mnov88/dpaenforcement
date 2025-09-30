import argparse
import sys
import types

sys.modules.setdefault("shap", types.ModuleType("shap"))

import pytest

pd = pytest.importorskip("pandas")

from scripts.evenness import cli
from scripts.evenness.omniscan import OmniScanOutputs


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


def test_cmd_omniscan_invokes_runner(monkeypatch, capsys):
    called = {}

    dummy = OmniScanOutputs(
        feature_universe_json="features.json",
        coverage_ledger_csv="coverage.csv",
        coverage_checklist_csv="checklist.csv",
        importance_heatmap_csv="importance.csv",
        interaction_map_csv="interactions.csv",
        block_importance_csv="blocks.csv",
        shap_country_csv="country.csv",
        shap_dpa_csv="dpa.csv",
        sage_importance_csv="sage.csv",
        specification_curve_csv="spec.csv",
        stability_selection_csv="stability.csv",
        knockoff_results_csv="knockoff.csv",
        robust_driver_csv="drivers.csv",
        crt_results_csv="crt.csv",
        jurisdiction_effects_csv="jurisdictions.csv",
        heterogeneity_csv="heterogeneity.csv",
        network_edges_csv="network.csv",
        community_summary_csv="communities.csv",
        risk_band_parity_csv="risk.csv",
        distribution_contrasts_csv="dist.csv",
    )

    def fake_run(paths, **kwargs):
        called["paths"] = paths
        return dummy

    monkeypatch.setattr(cli, "run_omniscan", fake_run)
    cli.cmd_omniscan(argparse.Namespace())
    out = capsys.readouterr().out
    assert "Omni-scan artefacts generated" in out
    assert "coverage.csv" in out
    assert "paths" in called


def test_build_parser_registers_phase_zero():
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert "phase-zero" in subcommands
