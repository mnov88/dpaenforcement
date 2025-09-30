# Scripts Reference and Pipeline Guide

This document summarises the ingestion, cleaning, and analysis scripts in this repository. It explains what each module does, its primary inputs/outputs (with usage examples), highlights heavy dependencies, and notes current limitations.

## Core cleaning & QA pipeline (`scripts/`)

1. **Generate enum metadata**
   ```bash
   python -m scripts.cli build-enum-whitelist \
     --prompt-path analyzed-decisions/data-extraction-prompt-sent-to-ai.md \
     --out resources/enum_whitelist.json
   ```
2. **Clean the wide dataset**
   ```bash
   python -m scripts.cli clean-wide \
     --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
     --out-csv outputs/cleaned_wide.csv \
     --validation-report outputs/validation_report.json
   ```
3. **Reconcile fines with human annotations (default in `run-all`)**
   ```bash
   python -m scripts.cli reconcile-fines \
     --wide-csv outputs/cleaned_wide.csv \
     --out-csv outputs/cleaned_wide_with_human_overrides.csv
   ```
   The command writes the reconciled dataset and can emit diagnostics with
   `--comparison-csv` / `--summary-json`. `run-all` executes this step automatically;
   add `--skip-fine-reconciliation` to opt out or `--reconciled-out-csv` to control
   the destination file.
4. **Emit long tables**
   ```bash
   python -m scripts.cli emit-long \
     --input-csv outputs/cleaned_wide.csv \
     --out-dir outputs/long_tables
   ```
   The emitter now writes `isic_assignments.csv`, capturing every parsed sector code (section/division/group/class) plus unmatched tokens alongside the reference version hash for downstream joins.
   Adjust `--input-csv` if you wrote the reconciled data to a separate path.
5. **Run consistency checks / QA summaries**
   ```bash
   python -m scripts.cli consistency --input-csv outputs/cleaned_wide.csv --report-json outputs/consistency_report.json
   python -m scripts.cli qa-summary --wide-csv outputs/cleaned_wide.csv --out-csv outputs/qa_summary.csv
   ```
6. **One-shot orchestration** – `python -m scripts.cli run-all` performs steps 1–5 using defaults from within the repo.
   - Extend the workflow with `--build-feature-matrix` (plus optional `--feature-matrix-parquet` / `--feature-matrix-metadata`) to materialise analysis artefacts, and `--run-evenness` to launch Phases 0–3 in sequence. Control the evenness working copy with `--evenness-wide-csv` (default `outputs/cleaned_wide_latest.csv`), set `--evenness-light` to skip SHAP/SAGE/knockoffs for laptop-friendly Phase 0 runs, and enable GPU acceleration via `--evenness-use-gpu` when drivers are available.

Configuration defaults are in `scripts/config.yaml`; see code comments for optional parameters.

## Omni-scan & evenness toolkit (`scripts/evenness/`)

The `scripts/evenness/` package implements the Phase 0 → Phase 3 programme documented in `research/phase0_omniscan.md`.

### Heavy dependencies
- `lightgbm`, `catboost`, `shap`, `numba`, `networkx[default]`, `graphviz`, `seaborn`. Install before running the CLI (e.g., `python -m pip install lightgbm catboost shap networkx seaborn`).
- Some steps (SHAP, knockoffs) are computationally expensive; run Phase 0 on a workstation-class machine.

### CLI overview (`scripts/evenness/cli.py`)

| Command | Purpose | Output directory |
|---------|---------|------------------|
| `python -m scripts.evenness.cli phase-zero` | Phase 0 omni-scan (feature universe, coverage ledgers, baseline drivers, fairness diagnostics) | `outputs/evenness/omniscan/` |
| `python -m scripts.evenness.cli phase-one` | Phase 1 matching & balanced factual matrices (generates `X_full`, `X_timeobs`, matching edges) | `outputs/evenness/` |
| `python -m scripts.evenness.cli phase-two` | Phase 2 uniformity tests (residuals, calibration, parity metrics) | `outputs/evenness/uniformity/` |
| `python -m scripts.evenness.cli phase-three --outcome fine_log1p` | Phase 3 explanation & policy (driver leaderboard, decomposition, policy levers, robustness) | `outputs/evenness/phase_three/` |
| `python -m scripts.evenness.cli fit-models` | Conditional disparity OLS/logit/mixed models | `outputs/evenness/models/` |
| `python -m scripts.evenness.cli robustness` | Specification curves, stability selection, knockoffs | `outputs/evenness/robustness/` |
| `python -m scripts.evenness.cli interaction-scan` | Jurisdiction-driver interactions | `outputs/evenness/models/interaction_scan.csv` |
| `python -m scripts.evenness.cli predictive --outcome fine_log1p` | Gradient boosting diagnostics + SHAP plots | `outputs/evenness/models/` |

See `scripts/evenness/config.py` for the full list of artefact paths (`EvennessPaths`).

### Phase 0 outputs (when you have the hardware)
Running `phase-zero` writes:
- Feature universe & coverage: `features_universe.json`, `coverage_ledger.csv`, `no_feature_left_behind.csv`.
- Importance diagnostics: `importance_heatmap.csv`, `interaction_map.csv`, `block_importance.csv`, `shap_country_summary.csv`, `shap_dpa_summary.csv`, `sage_importance.csv`.
- Stability & fairness: `specification_curve.csv`, `stability_selection.csv`, `knockoff_results.csv`, `robust_driver_list.csv`, `crt_results.csv`, `jurisdiction_effects.csv`, `heterogeneity_map.csv`.
- Risk-band & network analysis: `risk_band_parity.csv`, `risk_band_distribution.csv`, `network_edges.csv`, `network_communities.csv`.

If the full run is impractical locally, coordinate with a collaborator to execute Phase 0 and share the resulting CSV/JSON files for review.

## Breach-notification analysis (`scripts/analysis/`)

- `build_feature_matrix.py` – materialises ML-ready features (aggregated counts + latent components + per-token powers). Output: `outputs/analysis/feature_matrix.parquet` and `feature_matrix_metadata.json`.
  - Expands `isic_section` into `ISIC_SECTION_*` binaries (including a missingness flag) and adds high-frequency `ISIC_DIVISION_*` indicators so the evenness toolkit can consume sector hierarchy information directly.
- `run_diagnostics.py` – summarises notification flags, power bundles, and coverage statistics (`outputs/analysis/diagnostics/`).
- `hierarchical_severity_model.py` / `hierarchical_severity_regularized.py` – severity classification models.
- `joint_notification_sanction.py` / `joint_notification_bootstrap.py` – propensity/AIPW estimation with trimming & bootstrapping.
- `interaction_latent_analysis.py` – co-occurrence networks and latent PCA scores (`outputs/analysis/interaction/`).
- `generate_notification_report.py` – builds the HTML report with figures (`outputs/analysis/report/`).

## Known gaps / TODOs
- Resource path detection still relies on relative paths; refactor to use repo-root discovery.
- Omni-scan runtime is heavy; consider implementing sampling switches or an “outcome allowlist” to support lighter runs.
- Severity models lack uncertainty estimates; add bootstraps or Bayesian analogues.
- Notification AIPW remains sensitive to overlap; investigate overlap weighting or targeted learning.

Use this document as the practical guide when extending scripts or coordinating with collaborators on heavy analyses.
