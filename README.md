# DPA Enforcement Data Pipeline

This repository organizes raw and AI-annotated GDPR enforcement decisions, providing a reproducible pipeline for parsing, cleaning, exploratory diagnostics, and policy analysis. The workflow now spans:

- **Phase 0 (Omni-scan)** – feature-universe expansion and baseline diagnostics (`python -m scripts.evenness.cli phase-zero`).
- **Phase 1–3** – matching, uniformity checks, and notification/policy modelling (see `scripts/evenness/cli.py`).

## Repository layout
- `raw-data/` – source CSVs and machine translations; treat as append-only inputs. The merged canonical feed currently lives at `raw-data/LATEST_MASTER_ONLY_USE_THIS_MERGED.csv` and supersedes older analyzed extracts when its record count is higher.
- `analyzed-decisions/` – AI-extracted responses including the authoritative `master-analyzed-data-unclean.csv`.
- `scripts/` – CLI entry points and cleaning/analysis modules configured via `scripts/config.yaml` and described in `scripts-readme.md`.
- `outputs/` – generated artifacts (cleaned tables, validation reports, long tables, analysis reports); never edit by hand.
- `resources/` – lookup tables (ISIC codes, enum whitelists) consumed by the pipeline.
- `research/` – notebooks and memos (e.g., `breach_notification_prelim.md`, `phase0_omniscan.md`).
- `outputs/analysis/` – breach-notification reports, diagnostics, latent scores (`feature_matrix.parquet`, HTML outputs, etc.).
- `outputs/evenness/` – Phase 0/1/2/3 artefacts (`omniscan`, `phase_three`, etc.).

## Workflow quick start
1. **Build enum metadata**
   ```bash
   python -m scripts.cli build-enum-whitelist \
     --prompt-path analyzed-decisions/data-extraction-prompt-sent-to-ai.md \
     --out resources/enum_whitelist.json
   ```
2. **Clean the wide dataset**
   ```bash
   python -m scripts.cli clean-wide \
     --input-csv raw-data/LATEST_MASTER_ONLY_USE_THIS_MERGED.csv \
     --out-csv outputs/cleaned_wide.csv \
     --validation-report outputs/validation_report.json
   ```
   > When the merged source shrinks or older analyzed decisions need inspection, you can still point `--input-csv` at `analyzed-decisions/master-analyzed-data-unclean.csv`. By default, prefer the merged file whenever its decision count is larger.
   > The `run-all` orchestration now applies human-annotated fine overrides immediately after this step. You can opt out with `--skip-fine-reconciliation`, or capture the comparison artefacts via `--fine-comparison-csv` / `--fine-summary-json`.
   > Add `--build-feature-matrix` (optionally overriding destinations via `--feature-matrix-parquet` / `--feature-matrix-metadata`) to materialise analysis datasets, and `--run-evenness` to trigger Phases 0–3 automatically. Use `--evenness-wide-csv` to control the working copy (defaults to `outputs/cleaned_wide_latest.csv`), `--evenness-light` to skip SHAP/SAGE/knockoffs for quicker laptop runs, and `--evenness-use-gpu` when hardware is available.
3. **Phase 0 omni-scan (optional but recommended with a beefy machine)**
   ```bash
   cp outputs/cleaned_wide.csv outputs/cleaned_wide_latest.csv
   python -m scripts.evenness.cli phase-zero
   # Optional GPU acceleration for supported learners (LightGBM/CatBoost):
   python -m scripts.evenness.cli phase-zero --gpu
   # Lightweight laptop run (skips SHAP/SAGE/knockoffs):
   python -m scripts.evenness.cli phase-zero --light
   ```
   This produces feature coverage ledgers, baseline drivers, knockoff/stability results, and fairness diagnostics under `outputs/evenness/omniscan/`.
   A rotating log is written to `outputs/evenness/omniscan/phase0_run.log`.
4. **Emit long tables / run targeted analyses**
   ```bash
   python -m scripts.cli emit-long \
     --input-csv outputs/cleaned_wide.csv \
     --out-dir outputs/long_tables
   ```
   For a dedicated reconciliation pass (outside `run-all`), call:
   ```bash
   python -m scripts.cli reconcile-fines \
     --wide-csv outputs/cleaned_wide.csv \
     --out-csv outputs/cleaned_wide_with_human_overrides.csv
   ```
   Add `--comparison-csv` / `--summary-json` to persist diagnostics for review.
5. **Breach-notification analysis & reporting**
   - Feature matrix + latent components: `outputs/analysis/feature_matrix.parquet`, `latent_scores.parquet`.
   - Report generator: `python -m scripts.analysis.generate_notification_report` → `outputs/analysis/report/breach_notification_report.html`.

Tests can be run end-to-end with `pytest`. Temporary fixtures should live under `.tmp_*` directories to avoid polluting data folders.

## Evenness toolkit quick notes
- `python -m scripts.evenness.cli phase-one` / `phase-two` / `phase-three` run the matching, uniformity, and explanation/policy phases on top of Phase 0 outputs.
- `python -m scripts.evenness.cli fit-models`, `robustness`, `interaction-scan`, `predictive`, etc., give advanced diagnostics (conditional models, specification curves, interaction sweeps).

## Evenness Phase 0 (Omni-scan) – heavy workflow
If you (or a collaborator) run Phase 0 on a workstation with adequate resources:
- Ensure the feature sanitiser change (safe column names) is present.
- Expect long runtimes; SHAP, SAGE, and knockoffs are computationally intensive.
- Share the resulting CSV/JSON files so others can inspect coverage and drivers without rerunning the job.

## Additional documentation
- `scripts-readme.md` – in-depth module documentation and outstanding technical debt.
- `data-cleaning.md` – historical plan for transitioning from raw responses to structured tables.
- `phase0_omniscan.md` – design notes for the omni-scan implementation.
- `AGENTS.md` – contributor guidelines for tooling, data handling, and review processes.
