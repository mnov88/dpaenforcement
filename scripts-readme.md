# Scripts Reference and Pipeline Guide

This document summarizes every script that powers the DPA decision ingestion and cleaning workflow. It explains what each module does, its primary inputs and outputs (with short usage examples), outlines known limitations, and provides a recommended end-to-end run order.

## High-priority fixes identified

1. ✅ **`run-all` raw-source coupling** – `run-all` now feeds the cleaned wide CSV into the long-table and consistency stages by default, while the `emit-long`/`consistency` commands accept an `--input-format` flag (or auto-detect) so callers can still target the raw file when needed.【F:scripts/cli.py†L62-L108】【F:scripts/clean/long_tables.py†L34-L130】【F:scripts/clean/consistency.py†L12-L93】
2. ✅ **Enum whitelist metadata** – `build_enum_whitelist` records the actual prompt path in the emitted metadata, keeping provenance accurate when alternative questionnaires are parsed.【F:scripts/parser/enums.py†L23-L49】【F:scripts/cli.py†L25-L31】
3. **Resource path resolution** – Multiple cleaning modules resolve optional assets (enum whitelist, ISIC structure) using bare relative paths such as `Path("resources/…")`, which breaks when the CLI runs outside the repository root.【F:scripts/clean/long_tables.py†L20-L28】【F:scripts/clean/wide_output.py†L92-L101】
   - *Plan*: centralize resource discovery (e.g., via `Path(__file__).resolve().parents[2]`) or allow explicit CLI arguments so callers can supply absolute paths; add regression tests that invoke the CLI from a temporary working directory.

## Pipeline quick start

1. **Generate or refresh the enum whitelist** – required before validating multi-select tokens.
   ```bash
   python -m scripts.cli build-enum-whitelist \
     --prompt-path analyzed-decisions/data-extraction-prompt-sent-to-ai.md \
     --out resources/enum_whitelist.json
   ```
2. **Clean the wide dataset** – parses raw questionnaire responses, normalizes fields, and writes validation flags.
   ```bash
   python -m scripts.cli clean-wide \
     --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
     --out-csv outputs/cleaned_wide.csv \
     --validation-report outputs/validation_report.json
   ```
3. **Emit long-form tables** – expand multi-select answers into tidy per-option CSVs for downstream analysis (auto-detects whether the source is raw or cleaned wide).
  ```bash
  python -m scripts.cli emit-long \
    --input-csv outputs/cleaned_wide.csv \
    --out-dir outputs/long_tables
  ```
4. **Run consistency checks** – flag logical conflicts across questionnaire answers, reusing the cleaned wide dataset by default.
  ```bash
  python -m scripts.cli consistency \
    --input-csv outputs/cleaned_wide.csv \
    --report-json outputs/consistency_report.json
  ```
5. **Produce QA coverage summary** – aggregate unknown tokens and status coverage per multi-select.
   ```bash
   python -m scripts.cli qa-summary \
     --wide-csv outputs/cleaned_wide.csv \
     --out-csv outputs/qa_summary.csv
   ```
6. **(Optional) One-shot orchestration** – `python -m scripts.cli run-all` performs steps 1–5 using default paths from within the repository.【F:scripts/cli.py†L15-L114】

Configuration defaults for the above paths are also tracked in `scripts/config.yaml` for reference or external tooling.【F:scripts/config.yaml†L1-L16】

## Module catalog

### CLI orchestration (`scripts/cli.py`)

| Subcommand | Purpose | Key inputs | Outputs / side effects | Example |
|------------|---------|------------|-------------------------|---------|
| `build-enum-whitelist` | Parse the questionnaire prompt to create enum/multi-select specifications.【F:scripts/cli.py†L25-L33】 | Markdown prompt file | JSON whitelist written to disk | `python -m scripts.cli build-enum-whitelist --prompt-path … --out …` |
| `parse-stdin` | Quickly parse concatenated `Answer N:` blocks from STDIN and stream JSON lines for smoke testing.【F:scripts/cli.py†L36-L50】 | Raw text piped via STDIN | JSON per record to STDOUT | `cat raw.txt | python -m scripts.cli parse-stdin` |
| `clean-wide` | Produce the normalized wide CSV and validation report.【F:scripts/cli.py†L53-L59】 | Raw analysis CSV (`response` column) | Wide CSV + validation JSON written to paths | `python -m scripts.cli clean-wide --input-csv …` |
| `emit-long` | Expand multi-select answers into per-option long tables under a directory (auto-detects raw vs wide schema or accept `--input-format`).【F:scripts/cli.py†L62-L68】【F:scripts/clean/long_tables.py†L34-L130】 | Raw or cleaned wide CSV | `outputs/long_tables/*.csv` | `python -m scripts.cli emit-long --input-csv outputs/cleaned_wide.csv --out-dir …` |
| `consistency` | Run logical QA checks and emit a JSON report over raw or cleaned inputs.【F:scripts/cli.py†L71-L76】【F:scripts/clean/consistency.py†L12-L93】 | Raw or cleaned wide CSV | JSON report | `python -m scripts.cli consistency --input-csv outputs/cleaned_wide.csv` |
| `qa-summary` | Aggregate status/unknown token statistics across wide CSV columns.【F:scripts/cli.py†L79-L84】 | Cleaned wide CSV | Summary CSV | `python -m scripts.cli qa-summary --wide-csv …` |
| `run-all` | Orchestrate the full pipeline with overrideable defaults (passes the cleaned wide CSV to downstream steps by default; override with `--long-input-format raw` or `--consistency-input-format raw` if needed).【F:scripts/cli.py†L87-L114】 | Optional overrides for prompt, CSVs, and outputs | Regenerates all downstream artefacts | `python -m scripts.cli run-all --out-csv custom.csv` |

**Notable gaps / risks**
- Long-table and consistency exports depend on the `raw_qXX` columns emitted by the wide cleaner; update both modules together when adding/removing questions from the export set.【F:scripts/clean/wide_output.py†L40-L135】【F:scripts/clean/long_tables.py†L34-L130】【F:scripts/clean/consistency.py†L12-L93】
- Enum whitelist generation assumes the prompt follows the `**Answer N:**` pattern; deviations will silently fall back to tagging the spec as `RAW`.【F:scripts/parser/enums.py†L23-L41】

### Parser package (`scripts/parser`)

#### `ingest.py`
- **Functionality**: Splits concatenated response blocks at each `Answer 1:` marker and parses `Answer N:` lines into `QNN` keys, capturing completeness metadata.【F:scripts/parser/ingest.py†L7-L55】
- **Inputs**: Raw text block per decision; expected to contain exactly 68 numbered answers (`EXPECTED_QUESTION_COUNT`).【F:scripts/parser/ingest.py†L4-L21】 Example usage:
  ```python
  from scripts.parser.ingest import parse_record
  record = parse_record('Answer 1: ISO_3166-1_ALPHA-2:FR\nAnswer 2: CNIL\n…')
  ```
- **Outputs**: Dict with `answers` (`Q1`…`Q68` strings) and `metadata` (line count, completeness, missing questions, parser version).【F:scripts/parser/ingest.py†L37-L55】
- **Gaps**: Does not validate duplicate answers or enforce ordering; schema-echo cleanup happens later in the cleaning stage.

#### `enums.py`
- **Functionality**: Extract enum/multi-select/type specifications from the questionnaire prompt using regex, producing a whitelist map used during validation (including provenance metadata for the prompt path).【F:scripts/parser/enums.py†L4-L49】【F:scripts/cli.py†L25-L33】
- **Inputs/Outputs**: Markdown prompt text (+ optional `source_path`) → JSON-ready dict (`questions`, `source`). Example call: `build_enum_whitelist(prompt_text, source_path=str(prompt_path))`.
- **Gaps**: Lines that do not immediately follow the regex pattern are labeled `RAW`, so nested formatting (e.g., bullet lists) may require manual cleanup before ingestion.【F:scripts/parser/enums.py†L31-L41】

#### `validators.py`
- **Functionality**: Helper utilities for whitespace normalization and order-preserving de-duplication used during long-table emission.【F:scripts/parser/validators.py†L4-L15】
- **Gaps**: No direct CLI exposure; additional validation (e.g., token canonicalization) must be handled by downstream modules.

### Cleaning utilities (`scripts/clean`)

#### `typing_status.py`
- **Functionality**: Centralizes typed parsing for dates, countries, numeric fields, and multi-select status derivation, including schema-token stripping and exclusivity checks.【F:scripts/clean/typing_status.py†L29-L205】
- **Inputs/Outputs**:
  - `parse_date_field('2023-05-15')` → `DateParseResult(value=datetime(...), status='DISCUSSED')`.
  - `parse_number('€1 200,50')` → `NumericParseResult(value=1200.5, status='DISCUSSED')` after sanitization.【F:scripts/clean/typing_status.py†L101-L145】
  - `split_multiselect('A, B, NOT_APPLICABLE')` → token list with derived status via `derive_multiselect_status`.【F:scripts/clean/typing_status.py†L154-L205】
- **Gaps**:
  - Negative numeric values are rejected outright (`NEGATIVE_VALUE`), which may need adjustment for contexts like penalties reductions.【F:scripts/clean/typing_status.py†L136-L144】
  - Country normalization simply strips the ISO prefix and lacks alias mapping (e.g., `UK` handled later in `wide_output`).【F:scripts/clean/typing_status.py†L42-L53】

#### `enum_validate.py`
- **Functionality**: Loads the whitelist JSON and validates multi-select tokens, returning known vs unknown splits for downstream reporting.【F:scripts/clean/enum_validate.py†L8-L33】
- **Inputs/Outputs**: `EnumWhitelist.load(Path('resources/enum_whitelist.json'))` and `validate_tokens('Q30', tokens)`.
- **Gaps**: `load` assumes the JSON file exists; callers must guard with `Path.exists()` as done in `wide_output`/`long_tables`.

#### `isic_map.py`
- **Functionality**: Lightweight index over the ISIC Rev.4 structure to map activity codes to descriptions and sections.【F:scripts/clean/isic_map.py†L20-L55】
- **Inputs/Outputs**: Text/CSV resource → `IsicIndex`; `lookup('6209')` returns `(IsicEntry(...), True)` or gracefully falls back to 2-digit divisions.【F:scripts/clean/isic_map.py†L40-L55】
- **Gaps**: Assumes the resource file is CSV-like with quoted fields; no caching between runs; absent file quietly disables ISIC enrichment (handled by `wide_output`).【F:scripts/clean/wide_output.py†L80-L83】

#### `geo_enrich.py`
- **Functionality**: Classifies countries into EU/EEA groupings and placeholders for DPA canonical names.【F:scripts/clean/geo_enrich.py†L5-L27】
- **Inputs/Outputs**: `enrich_country('FR')` → `('FR', 'EU')`; `normalize_dpa_name('CNIL')` currently returns the raw value.
- **Gaps**: Country lists are minimal and static; DPA normalization is a stub awaiting registry integration.【F:scripts/clean/geo_enrich.py†L5-L27】

#### `text_norm.py`
- **Functionality**: Unicode normalization, zero-width stripping, whitespace collapsing, and a very lightweight language heuristic for key narrative answers.【F:scripts/clean/text_norm.py†L17-L36】
- **Inputs/Outputs**: `normalize_text('\u200bHello  world')` → `('Hello world', tokens=2, chars=11)`; `detect_language_heuristic('Hello')` → `'ENGLISH_LIKELY'` for high ASCII ratios.【F:scripts/clean/text_norm.py†L17-L36】
- **Gaps**: Language heuristic only distinguishes ASCII-dominant vs other; consider integrating a proper language detector for multilingual content.

#### `wide_output.py`
- **Functionality**: Core cleaner that reads the annotated CSV, re-parses raw responses, normalizes geography, dates, numerics, text, and multi-selects, then writes a wide table plus a validation report (including `raw_qXX` columns for downstream reuse).【F:scripts/clean/wide_output.py†L76-L310】
- **Inputs**: Raw CSV with `ID` and `response` columns; optional whitelist and ISIC resources if present under `resources/` (detected dynamically).【F:scripts/clean/wide_output.py†L80-L144】 Example invocation via CLI shown earlier.
- **Outputs**:
  - Wide CSV containing ingest metadata, normalized fields, derived indicators, per-option boolean flags, schema-echo tracking, and raw answer exports for consistency/long-table reuse.【F:scripts/clean/wide_output.py†L89-L299】
  - Validation JSON storing per-decision QA flags (fine amount issues, parse errors).【F:scripts/clean/wide_output.py†L300-L310】
- **Gaps**:
  - Multi-select coverage relies on whitelist tokens; if the whitelist is empty the script still writes coverage columns but cannot flag unknowns.【F:scripts/clean/wide_output.py†L114-L279】
  - Severity, remedy-only, and derived counts depend on exact token strings; upstream schema changes require whitelist updates to remain accurate.【F:scripts/clean/wide_output.py†L288-L296】
  - Schema-echo detection only checks for three prefixes; additional artefacts (e.g., `LIST:`) will pass through until manually added.【F:scripts/clean/wide_output.py†L63-L158】

#### `long_tables.py`
- **Functionality**: Builds tidy per-question tables by expanding multi-select answers and tagging unknown tokens for review, whether sourced from the raw CSV or the cleaned wide export.【F:scripts/clean/long_tables.py†L20-L130】
- **Inputs/Outputs**: Raw or cleaned wide CSV (`emit_from_csv` auto-detects unless overridden) → multiple CSVs such as `article_5_discussed.csv`, `breach_types.csv`, etc.【F:scripts/clean/long_tables.py†L47-L130】 Example call:
  ```python
  from scripts.clean.long_tables import LongEmitter
  LongEmitter(Path('outputs/long_tables')).emit_from_csv(Path('outputs/cleaned_wide.csv'), input_format='wide')
  ```
- **Gaps**:
  - Empty multi-selects emit status-only rows but do not capture explicit `NOT_MENTIONED` counts, so downstream completeness metrics must use the wide CSV coverage columns instead.【F:scripts/clean/long_tables.py†L73-L88】

#### `consistency.py`
- **Functionality**: Applies rule-based QA checks across related questions (breach chain, fine vs. enforcement, appeal logic, turnover caps) and records flags per decision, regardless of whether the input is raw or cleaned wide (leveraging the exported `raw_qXX` columns).【F:scripts/clean/consistency.py†L12-L93】
- **Inputs/Outputs**: Raw or cleaned wide CSV → JSON array of `{decision_id, flags}` objects for flagged records.【F:scripts/clean/consistency.py†L12-L93】
- **Gaps**:
  - Geo verification for cross-border claims is best-effort; it only inspects question text endings like `": EU"` and does not consult the enriched country grouping from the wide output.【F:scripts/clean/consistency.py†L64-L73】
  - Additional legal cross-checks (Article 33/34 interplay, turnover cap math) are noted as TODOs in the top-level scripts README.

#### `qa_summary.py`
- **Functionality**: Tallies per-question coverage statistics (rows with known/unknown tokens, status distributions, top unknown tokens) across the cleaned wide CSV.【F:scripts/clean/qa_summary.py†L30-L104】
- **Inputs/Outputs**: Wide CSV produced by `clean_csv_to_wide`; summary CSV with serialized JSON columns for quick inspection or dashboarding.【F:scripts/clean/qa_summary.py†L30-L104】
- **Gaps**:
  - Skips fields whose expected columns are absent; a misaligned wide CSV schema will silently drop sections from the summary.【F:scripts/clean/qa_summary.py†L40-L48】
  - Relies on the prefixes staying in sync with `MULTI_FIELDS` in `wide_output.py`; any divergence must be updated in both files.【F:scripts/clean/qa_summary.py†L9-L27】

### Package markers and configuration
- `scripts/__init__.py` and `scripts/clean/__init__.py` expose the packages for external imports; they contain descriptive docstrings only.【F:scripts/__init__.py†L1-L2】【F:scripts/clean/__init__.py†L1-L1】
- `scripts/config.yaml` centralizes default paths and options (strict enum enforcement, warning logging) for use by orchestration tooling; it is informational today because the Python modules accept CLI parameters directly.【F:scripts/config.yaml†L1-L16】 A future enhancement could load this configuration automatically.

## Known cross-module gaps
- **Schema echo handling**: Multiple modules use the same tuple of schema prefixes; consider centralizing this constant to avoid drift when new tokens appear.【F:scripts/clean/wide_output.py†L33-L65】【F:scripts/clean/long_tables.py†L12-L17】
- **Resource dependencies**: Optional assets (enum whitelist, ISIC mapping) are loaded only if the files exist, which can lead to silent degradation. Automating integrity checks before the pipeline runs would improve reproducibility.【F:scripts/clean/wide_output.py†L80-L144】【F:scripts/clean/long_tables.py†L20-L45】
- **Configuration file usage**: `scripts/config.yaml` is not programmatically consumed, so downstream automation must keep CLI flags and config in sync manually.【F:scripts/config.yaml†L1-L16】

Use this guide as the authoritative reference when extending the ingestion/cleaning scripts or wiring them into scheduled jobs.
