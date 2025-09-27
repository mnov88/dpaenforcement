# Methodological review of data cleaning and setup scripts

## 1. Alignment with project goals and specification
The cleaning pipeline closely follows the guiding principles documented in `data-cleaning.md`, prioritising preservation of legal nuance, reproducibility, and the creation of both wide and relational outputs.【F:data-cleaning.md†L1-L200】 The implementation exposes separate modules for ingestion (`scripts/parser`), typing and status handling (`scripts/clean/typing_status.py`), enrichment (`scripts/clean/geo_enrich.py`), and tabular reshaping (`scripts/clean/wide_output.py`, `scripts/clean/long_tables.py`). This modular structure supports auditability, makes the transformations reproducible, and mirrors the multi-table deliverables outlined in the specification.

## 2. Ingestion and parsing logic
`parse_record` and `segment_records` implement the required splitting of concatenated response blocks and extraction of `Answer N:` tokens using anchored regular expressions, while tracking completeness metadata, missing questions, and parser warnings.【F:scripts/parser/ingest.py†L1-L56】 The logic ensures that every decision carries parser versioning and completeness diagnostics, which is essential for downstream auditing.

### Observed strengths
- Record segmentation explicitly retains only blocks that begin with `Answer 1:`, preventing bleed-over between decisions.【F:scripts/parser/ingest.py†L11-L22】
- Metadata captures counts and missing questions as required by the spec, enabling validation reports and reproducibility.【F:scripts/parser/ingest.py†L38-L55】

### Potential enhancements
- Source identifiers (`source_id`, ingestion timestamps) mentioned in the spec are not yet emitted. Adding them to `metadata` would further support lineage tracking.

## 3. Typing, missingness, and numeric handling
The `typing_status` module provides dataclasses for date, numeric, and multi-select parsing, embedding status flags that distinguish `NOT_MENTIONED`, `NOT_APPLICABLE`, parse errors, and other legally significant states.【F:scripts/clean/typing_status.py†L9-L198】 It strips schema echo tokens before numeric parsing, catches negative values, and flags parsing errors without silently coercing them—consistent with the legal caution emphasised in the specification.【F:scripts/clean/typing_status.py†L101-L145】

### Recent fix
During this review we restored the missing imports and `UTC` handling so that the module executes correctly when imported. Without the fix, importing `typing_status` raised `NameError`/`SyntaxError`, blocking all downstream cleaners. The correction ensures date parsing now attaches a timezone-aware `datetime` and maintains the status taxonomy intact.【F:scripts/clean/typing_status.py†L1-L145】

### Recommendations
- Consider extending `_NUM_ALLOWED` sanitisation to recognise non-breaking spaces and locale-specific separators if new jurisdictions use them.
- The spec calls for additional derived temporal features (ISO week, duration metrics). These can be layered on top of the existing dataclasses without altering raw values.

## 4. Geographical and textual enrichment
`enrich_country` classifies ISO codes into EU/EEA/non-EEA groupings, while preserving unknowns, satisfying the project’s requirement to enable stratified regional analysis.【F:scripts/clean/geo_enrich.py†L5-L27】 Text fields are normalised to Unicode NFC, stripped of zero-width characters, and accompanied by light-language heuristics, preparing the narrative answers for qualitative analysis without erasing legally meaningful punctuation.【F:scripts/clean/text_norm.py†L1-L36】

The current DPA normaliser is a placeholder that simply trims text; the specification envisions a canonical registry with identifiers. Populating that lookup will be important for consistent aggregation across decisions.

## 5. Wide (flattened) decision table
`clean_csv_to_wide` constructs a single-row-per-decision table with extensive metadata, typed numbers, derived financial ratios, text normalisations, and a systematic set of multi-select indicators and status columns.【F:scripts/clean/wide_output.py†L76-L310】 Key methodological strengths include:

- Preservation of parser metadata (`ingest_*` columns) alongside cleaned fields, satisfying reproducibility goals.【F:scripts/clean/wide_output.py†L209-L217】
- Explicit numeric status/error columns, log transforms, outlier flags, and fine/turnover ratios for modelling readiness.【F:scripts/clean/wide_output.py†L165-L247】
- Automatic expansion of multi-selects into both coverage/status metadata and per-option boolean indicators, while logging exclusivity conflicts to highlight contradictory states like `NOT_APPLICABLE` plus substantive tokens.【F:scripts/clean/wide_output.py†L264-L297】
- Carrying normalised text plus token counts and heuristic language detection for the qualitative questions.【F:scripts/clean/wide_output.py†L200-L259】

The output thereby achieves the requested flattened structure with legally-aware missingness handling.

### Suggested refinements
- The current exclusivity logic flags contradictions but still encodes indicators as binary 0/1. Where exclusivity conflicts exist, adding a companion `_conflict` boolean per question could prevent analysts from mistakenly treating a `0` as evidence of absence.
- Additional derived counts for rights/access (beyond the three already emitted) would fully meet the roll-up metrics described in the spec’s section F.17-18.

## 6. Long-form relational tables
The `LongEmitter` exports relational tables for each multi-select question, labelling tokens as `KNOWN` or `UNKNOWN` relative to the whitelist and preserving status-only rows (e.g., `NONE_MENTIONED`).【F:scripts/clean/long_tables.py†L20-L126】 This fulfils the requirement to support graph/network analyses while keeping contradictory status codes in view.

One caution: tokens are deduplicated before deriving status. This matches the spec’s “deduplicate while preserving order,” but it also means repeated contradictory markers (e.g., multiple schema echoes) are removed before exclusivity analysis. If repeated contradictory tokens convey meaning in future data drops, consider deriving status from the raw token list prior to deduplication.

## 7. Consistency and validation checks
`run_consistency_checks` implements the cross-field validations outlined in section D of the spec, generating JSON reports that flag breach-notification inconsistencies, appeal logic mismatches, fine/turnover contradictions, and cross-border plausibility checks without mutating source answers.【F:data-cleaning.md†L105-L134】【F:scripts/clean/consistency.py†L12-L78】 This design respects the requirement to flag rather than auto-correct legal interpretations.

## 8. Outstanding gaps and priorities
1. **Canonical DPA registry and additional enrichment** – Implement the planned DPA lookup and extend country enrichment with language/region metadata for deeper stratified analyses.【F:data-cleaning.md†L137-L148】【F:scripts/clean/geo_enrich.py†L5-L27】
2. **Derived temporal metrics** – Add ISO-week and duration ordinals to align with the time features mandated in the specification.【F:data-cleaning.md†L151-L168】
3. **Comprehensive conflict signalling** – Augment multi-select outputs with explicit contradiction flags (e.g., `_status_conflict`) to complement the existing exclusivity indicator, ensuring analysts recognise status-only rows and conflicts at a glance.【F:scripts/clean/wide_output.py†L264-L297】【F:scripts/clean/long_tables.py†L73-L126】
4. **Metadata completeness** – Record ingestion timestamps/source IDs and propagate them to the flattened outputs to close the reproducibility loop suggested in the spec.【F:data-cleaning.md†L31-L35】【F:scripts/parser/ingest.py†L38-L55】

## 9. Conclusion
Overall, the current implementation operationalises the project’s methodological requirements: it preserves legally meaningful missingness, produces both wide and relational outputs, and surfaces logical contradictions without overwriting source evidence. The fixes applied during this review reinstate import integrity, ensuring the pipeline executes end-to-end. Addressing the remaining enrichment and conflict-annotation gaps will further align the codebase with the detailed roadmap outlined in `data-cleaning.md`.
