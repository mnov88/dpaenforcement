# Scripts

CLI utilities for ingestion, validation, and cleaning of 68‑answer decision records. The pipeline is strict on structure and non‑destructive on content: we normalize and flag, not overwrite meaning.

## Layout

- `scripts/cli.py`: CLI entrypoint with subcommands
- `scripts/parser/ingest.py`: Segment blocks and extract Answer N: value pairs
- `scripts/parser/enums.py`: Build ENUM/MULTI_SELECT/TYPE whitelist from the prompt
- `scripts/clean/wide_output.py`: Generate cleaned wide CSV with enrichment and derived fields
- `scripts/clean/long_tables.py`: Emit long‑form multi‑select tables
- `scripts/clean/consistency.py`: Cross‑field legal consistency checks
- `scripts/clean/isic_map.py`: ISIC Rev.4 index and lookup
- `scripts/clean/geo_enrich.py`: Country EU/EEA grouping and DPA canonicalization placeholder
- `scripts/clean/typing_status.py`: Typing, robust numeric parsing, and status helpers
- `scripts/clean/text_norm.py`: Text normalization (NFC, zero‑width removal, whitespace)
- `scripts/config.yaml`: Default paths and options (informational)

## Commands

- Build enum whitelist from the questionnaire prompt:

```bash
python3 -m scripts.cli build-enum-whitelist \
  --prompt-path analyzed-decisions/data-extraction-prompt-sent-to-ai.md \
  --out resources/enum_whitelist.json
```

- Parse concatenated Answer blocks from STDIN (smoke test):

```bash
cat some_raw_blocks.txt | python3 -m scripts.cli parse-stdin | head
```

- Produce cleaned wide CSV and validation report:

```bash
python3 -m scripts.cli clean-wide \
  --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
  --out-csv outputs/cleaned_wide.csv \
  --validation-report outputs/validation_report.json
```

- Emit long tables for multi‑selects:

```bash
python3 -m scripts.cli emit-long \
  --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
  --out-dir outputs/long_tables
```

- Run cross‑field consistency checks:

```bash
python3 -m scripts.cli consistency \
  --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
  --report-json outputs/consistency_report.json
```

- Run the full pipeline with defaults:

```bash
python3 -m scripts.cli run-all
```

## Outputs

- Wide CSV: `outputs/cleaned_wide.csv`
- Validation report (per‑decision minimal flags): `outputs/validation_report.json`
- Consistency report (legal cross‑checks): `outputs/consistency_report.json`
- Long tables: `outputs/long_tables/*.csv`

## Wide CSV: key fields

- Geography: `country_code`, `country_status`, `country_group` (EU/EEA/NON_EEA), `country_mapped_from_uk`, `country_whitelist_ok`.
- DPA: `dpa_name_raw`, `dpa_name_canonical` (placeholder for registry mapping).
- Dates: `decision_date`, `decision_date_status` (DISCUSSED/NOT_DISCUSSED), `decision_year`, `decision_quarter`.
- Numerics: `fine_eur`, `turnover_eur` with robust parsing (thousands/scientific; negatives rejected), `fine_log1p`, `turnover_log1p`, `fine_outlier_flag`, `turnover_outlier_flag`, `fine_to_turnover_ratio`.
- Article features: `legal_bases_discussed`, `principles_violated` (cleaned of schema‑echo tokens), counts (`n_principles_discussed`, `n_principles_violated`), measures (`n_corrective_measures`, `severity_measures_present`, `remedy_only_case`, `breach_case`).
- Sector: `isic_code`, `isic_desc`, `isic_section` (via `resources/ISIC_Rev_4_english_structure.txt`).
- Texts: normalized `q36_text`, `q52_text`, `q67_text`, `q68_text` with `*_lang` (heuristic) and `*_tokens`.
- Cleaning flags: `schema_echo_flag` and `schema_echo_fields` (semicolon‑joined list of affected questions).

## Long tables

Generated under `outputs/long_tables/`:
- `article_5_discussed.csv` (Q30), `article_5_violated.csv` (Q31), `article_6_discussed.csv` (Q32), `legal_basis_relied_on.csv` (Q33), `consent_issues.csv` (Q34), `li_test_outcome.csv` (Q35), `breach_types.csv` (Q21), `vulnerable_groups.csv` (Q46), `corrective_powers.csv` (Q53), `rights_discussed.csv` (Q56), `rights_violated.csv` (Q57).
- Schema‑echo tokens (e.g., `TYPE:...`, `ENUM:...`, `MULTI_SELECT:...`) are filtered.

## Consistency checks (selected)

- Admin fine requires non‑zero `Q37` amount.
- Breach chain sanity (subset): will be expanded to include Article 33/34 links, appeal logic, and caps vs turnover verification.

## Design principles

- Non‑destructive: keep raw values, add normalized fields and flags.
- Typed missingness: dates carry `decision_date_status`; multi‑selects keep status in long tables.
- Reproducibility: deterministic ordering, single CLI to run entire pipeline.

## Next enhancements (tracked as TODOs)

- DPA canonicalization registry, country whitelist enforcement, enhanced ISIC (multi‑code), extended legal checks, EDPB references parsing, quality score, strict mode and sampling, expanded tests, and data dictionary update.
