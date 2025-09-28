# DPA Enforcement Data Pipeline

This repository organizes raw and AI-annotated GDPR enforcement decisions and provides a reproducible pipeline for parsing, cleaning, and analyzing the dataset. The current focus is producing wide and long-form tables that support breach-evenness research and notification policy evaluation.

## Repository layout
- `raw-data/` – source CSVs and machine translations; treat as append-only inputs.
- `analyzed-decisions/` – AI-extracted question responses including the authoritative `master-analyzed-data-unclean.csv`.
- `scripts/` – CLI entry points and cleaning modules configured via `scripts/config.yaml` (see `scripts-readme.md` for a detailed catalog).
- `outputs/` – generated artifacts (cleaned tables, validation reports, evenness analysis outputs); never hand-edit.
- `resources/` – lookup tables (ISIC codes, enum whitelists) consumed by the pipeline.
- `tests/` – pytest coverage for parser, cleaners, and regression checks.
- `research/` – analyst notebooks and narrative reports summarizing breach-evenness findings.
- `phase3completed.patch` – snapshot patch capturing the latest phase-three notification estimator outputs for reproducibility reviews.

## Workflow quick start
1. Build enum metadata
   ```bash
   python -m scripts.cli build-enum-whitelist \
     --prompt-path analyzed-decisions/data-extraction-prompt-sent-to-ai.md \
     --out resources/enum_whitelist.json
   ```
2. Produce the cleaned wide CSV and validation report
   ```bash
   python -m scripts.cli clean-wide \
     --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
     --out-csv outputs/cleaned_wide.csv \
     --validation-report outputs/validation_report.json
   ```
3. Expand multi-select answers into tidy long tables
   ```bash
   python -m scripts.cli emit-long \
     --input-csv outputs/cleaned_wide.csv \
     --out-dir outputs/long_tables
   ```
4. Run consistency checks and QA summaries as needed
   ```bash
   python -m scripts.cli consistency --input-csv outputs/cleaned_wide.csv --report-json outputs/consistency_report.json
   python -m scripts.cli qa-summary --wide-csv outputs/cleaned_wide.csv --out-csv outputs/qa_summary.csv
   ```
5. Orchestrate everything with defaults
   ```bash
   python -m scripts.cli run-all
   ```

Tests can be run end-to-end with `pytest`. Temporary fixtures should live under `.tmp_*` directories so that data folders remain pristine.

## Evenness phase-three update
Recent work refines the phase-three notification estimator to better handle separation in match quality. The resulting artifacts live under `outputs/evenness/`, and their exact diff against the previous release is preserved in `phase3completed.patch`. Refer to `research/evenness_execution_report.md` and `research/evenness_policy_implications.md` for interpretation guidance.

## Additional documentation
- `scripts-readme.md` – in-depth module documentation and outstanding technical debt.
- `data-cleaning.md` – historical plan for transitioning from raw responses to structured tables.
- `AGENTS.md` – contributor guidelines for tooling, data handling, and review processes.
