# Repository Guidelines

## Project Structure & Module Organization
- `analyzed-decisions/` holds AI-extracted answers plus the authoritative `master-analyzed-data-unclean.csv`.
- `raw-data/` contains source CSVs and machine translations; treat as append-only.
- `scripts/` provides the Python pipeline (`cli.py`, `parser/`, `clean/`) configured via `scripts/config.yaml`.
- `outputs/` captures generated artifacts (cleaned CSVs, reports, long tables); never edit these manually.
- `resources/` stores lookup tables (e.g., ISIC descriptors) consumed by cleaning jobs.
- `tests/` contains regression coverage for parser and cleaning behaviors.

## Build, Test, and Development Commands
- Pipeline entrypoint: `python3 -m scripts.cli run-all` (runs parsing, cleaning, consistency checks with config defaults).
- Focused cleans: `python3 -m scripts.cli clean-wide --input-csv analyzed-decisions/master-analyzed-data-unclean.csv --out-csv outputs/cleaned_wide.csv --validation-report outputs/validation_report.json`.
- Long tables: `python3 -m scripts.cli emit-long --input-csv analyzed-decisions/master-analyzed-data-unclean.csv --out-dir outputs/long_tables`.
- Tests: `pytest` (auto-discovers unit tests under `tests/`).

## Coding Style & Naming Conventions
- Python 3 with 4-space indentation and type hints when interfaces are reused (`typing_status`, `isic_map`).
- Favor `pathlib.Path` and pure functions; keep I/O boundaries inside CLI handlers.
- Match modular layout: parser helpers in `scripts/parser`, cleaners in `scripts/clean`, shared utilities in `scripts/__init__.py`.
- Preserve upstream question identifiers (`Q1`–`Q68`) and enum tokens (e.g., `NOT_APPLICABLE`) verbatim; derive normalized fields separately.

## Testing Guidelines
- Extend `tests/test_parser_and_clean.py` or add new modules under `tests/` using `pytest`/`unittest` style.
- Use temporary directories (`Path(".tmp_*")`) for fixtures and clean them up to avoid polluting data dirs.
- When adding detectors, assert both valid parsing and expected status codes (`NOT_MENTIONED`, `MIXED_CONTRADICTORY`).

## Commit & Pull Request Guidelines
- Follow descriptive, sentence-style commit messages emphasizing data or pipeline impacts (see `git log` for precedent).
- Group commits by logical change (e.g., "Refine schema echo handling"), and include CSV/JSON diffs only when reproducible via scripts.
- Pull requests should summarize motivation, list generated outputs, reference related issues, and attach sample command/output snippets or screenshots when UI/data artifacts change.

## Data Handling Notes
- Never overwrite files in `raw-data/` or `analyzed-decisions/`; create derived data under `outputs/`.
- Document new configuration knobs inside `scripts/README.md` and keep `config.yaml` defaults conservative.
