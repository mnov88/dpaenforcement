# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DPA (Data Protection Authority) decision analysis project for GDPR compliance research. The project processes AI-extracted decision data from 68-field questionnaires into cleaned datasets for comparative and statistical analysis.

## Key Commands

### Data Pipeline
```bash
# Run the complete data processing pipeline
python3 -m scripts.cli run-all

# Build enum whitelist from questionnaire prompt
python3 -m scripts.cli build-enum-whitelist \
  --prompt-path analyzed-decisions/data-extraction-prompt-sent-to-ai.md \
  --out resources/enum_whitelist.json

# Clean wide dataset with validation
python3 -m scripts.cli clean-wide \
  --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
  --out-csv outputs/cleaned_wide.csv \
  --validation-report outputs/validation_report.json

# Generate long-form tables for multi-select fields
python3 -m scripts.cli emit-long \
  --input-csv outputs/cleaned_wide.csv \
  --out-dir outputs/long_tables

# Run consistency checks across fields
python3 -m scripts.cli consistency \
  --input-csv outputs/cleaned_wide.csv \
  --report-json outputs/consistency_report.json

# Generate QA coverage summary
python3 -m scripts.cli qa-summary \
  --wide-csv outputs/cleaned_wide.csv \
  --out-csv outputs/qa_summary.csv
```

### Advanced Export Formats
```bash
# Export to Parquet with partitioning (10x smaller, faster queries)
python3 -m scripts.cli export-parquet \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/parquet

# Export to Arrow/Feather for cross-language compatibility
python3 -m scripts.cli export-arrow \
  --wide-csv outputs/cleaned_wide.csv \
  --out-dir outputs/arrow \
  --compression zstd

# Export graph/network formats for relationship analysis
python3 -m scripts.cli export-graph \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/networks

# Export for statistical packages (R, Stata, SPSS)
python3 -m scripts.cli export-stats \
  --wide-csv outputs/cleaned_wide.csv \
  --out-dir outputs/statistical \
  --formats r,stata,spss

# Export ML-ready features with embeddings
python3 -m scripts.cli export-ml \
  --wide-csv outputs/cleaned_wide.csv \
  --out-dir outputs/ml_ready \
  --embeddings-model all-MiniLM-L6-v2 \
  --test-size 0.2
```

### Testing
```bash
# Run tests (pytest configuration in pytest.ini)
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/test_parser_and_clean.py
```

## Architecture

### Data Flow
1. **Raw Data**: `/raw-data` contains CSV with machine-translated decisions
2. **Analyzed Data**: `/analyzed-decisions` contains AI-processed 68-field responses
   - `master-analyzed-data-unclean.csv` is the primary input file
3. **Processing Scripts**: `/scripts` contains modular CLI utilities
4. **Outputs**: `/outputs` contains cleaned datasets and reports

### Core Modules

#### CLI (`scripts/cli.py`)
- Central orchestration with subcommands for each pipeline stage
- `run-all` command executes full pipeline with sensible defaults
- Auto-detects input formats (raw vs cleaned) for downstream stages

#### Parser Package (`scripts/parser/`)
- `ingest.py`: Parses concatenated "Answer N:" blocks into structured data
- `enums.py`: Extracts enum/multi-select specifications from questionnaire prompts
- `validators.py`: Text normalization and deduplication utilities

#### Cleaning Package (`scripts/clean/`)
- `wide_output.py`: Core cleaner producing normalized wide CSV with validation flags
- `long_tables.py`: Expands multi-select answers into tidy per-option tables
- `consistency.py`: Cross-field logical validation (fines, breach chains, appeals)
- `typing_status.py`: Centralized parsing for dates, countries, numerics, multi-selects
- `geo_enrich.py`: Country classification (EU/EEA groupings) and DPA normalization
- `text_norm.py`: Unicode normalization and language detection heuristics
- `isic_map.py`: Industry classification code lookup and enrichment
- `qa_summary.py`: Coverage statistics aggregation

#### Export Package (`scripts/export/`)
- `base.py`: Common infrastructure for all exporters with legal metadata preservation
- `parquet_export.py`: Apache Parquet format with partitioning and compression
- `arrow_export.py`: Arrow/Feather format for cross-language data science workflows
- `graph_export.py`: Network formats (GraphML, Neo4j CSV, NetworkX) for relationship analysis
- `stats_export.py`: Statistical packages (R RDS, Stata DTA, SPSS SAV) with proper factor levels
- `ml_export.py`: ML-ready features with text embeddings and stratified train/test splits

### Key Data Files
- `analyzed-decisions/master-analyzed-data-unclean.csv`: Primary input (ID + response columns)
- `analyzed-decisions/data-extraction-prompt-sent-to-ai.md`: Questionnaire specification
- `resources/enum_whitelist.json`: Generated enum validation specifications
- `resources/ISIC_Rev_4_english_structure.txt`: Industry classification reference
- `outputs/cleaned_wide.csv`: Normalized wide dataset (primary output)

### Design Principles
- **Non-destructive**: Raw values preserved alongside normalized fields
- **Typed missingness**: Status tracking for dates, multi-selects (DISCUSSED/NOT_DISCUSSED/etc.)
- **Reproducible**: Deterministic ordering, single CLI pipeline execution
- **Modular**: Each cleaning stage can run independently with explicit inputs/outputs

## Working with the Codebase

### Adding New Questions
1. Update enum specifications in questionnaire prompt
2. Regenerate whitelist: `python3 -m scripts.cli build-enum-whitelist`
3. Add parsing logic to relevant modules in `scripts/clean/`
4. Update `MULTI_FIELDS` in `wide_output.py` and corresponding modules
5. Test with sample data

### Resource Dependencies
- Enum whitelist and ISIC mapping are loaded conditionally (graceful degradation if missing)
- Run from repository root - modules use relative paths to `resources/`
- Default paths in `scripts/config.yaml` (informational reference)

### Schema Echo Handling
- AI responses may contain schema artifacts (`TYPE:`, `ENUM:`, `MULTI_SELECT:` prefixes)
- Stripped during cleaning with tracking in `schema_echo_flag` and `schema_echo_fields`
- Centralized in tuple constants across modules

### Advanced Export Formats

#### Parquet Export
- **Use case**: High-performance analytical queries, data science workflows
- **Benefits**: 10x file size reduction, columnar storage, direct pandas/DuckDB integration
- **Partitioning**: By country_group and decision_year for efficient filtering
- **Metadata**: Rich legal semantics preserved in Arrow schema

#### Arrow/Feather Export
- **Use case**: Cross-language compatibility (Python/R/Julia), zero-copy workflows
- **Benefits**: Memory-efficient, language interoperability, embedded metadata
- **Compression**: ZSTD compression for optimal size/speed trade-off

#### Graph/Network Exports
- **Formats**: GraphML (Gephi), GML (R igraph), Neo4j CSV, NetworkX
- **Networks**: DPA-Decision, Decision-Article bipartite, violation co-occurrence
- **Use cases**: Legal precedent analysis, enforcement pattern detection, centrality analysis

#### Statistical Package Exports
- **R**: RDS format with proper factor levels and variable labels
- **Stata**: DTA format with value labels and comprehensive metadata
- **SPSS**: SAV format optimized for policy research workflows
- **Features**: Codebooks, loading scripts, analysis templates included

#### ML-Ready Exports
- **Features**: 300+ engineered features preserving legal semantics
- **Embeddings**: Sentence transformer embeddings for narrative text fields
- **Splits**: Stratified train/test/validation preserving legal distributions
- **Templates**: Ready-to-use analysis scripts for common ML tasks

### Critical Data Recovery (2025-09-27)

**IMPORTANT**: A major data recovery was completed where 36 out of 68 questionnaire variables were previously missing from exports due to a limitation in `scripts/clean/wide_output.py`. The `RAW_QUESTION_EXPORT` tuple was updated to include all 68 questions, recovering critical timing variables:

- **Q15**: Investigation initiation (COMPLAINT, BREACH_NOTIFICATION, EX_OFFICIO_DPA_INITIATIVE)
- **Q19**: 72-hour timing compliance (YES_WITHIN_72H, NO_LATE, TIMING_DISPUTED)
- **Q20**: Delay amounts (1_TO_7_DAYS, 1_TO_4_WEEKS, 1_TO_6_MONTHS, OVER_6_MONTHS)

Plus 33 other variables including defendant profiles, breach characteristics, text fields, and legal significance markers. All exports now contain the complete 68-variable dataset enabling full academic analysis of GDPR breach notification compliance and enforcement patterns.

### Testing Strategy
- Tests focus on parser correctness and data integrity
- Use `test_parser_and_clean.py` for core functionality
- Pipeline outputs should be deterministic for regression testing
- Export formats validated against target software compatibility

### Dependencies for Advanced Exports
- **Optional**: `pyarrow` (Parquet/Arrow), `networkx` (graphs), `sentence-transformers` (embeddings)
- **Statistical packages**: `pyreadstat` (Stata/SPSS), `rpy2` (R integration)
- **All exporters**: Graceful degradation when optional dependencies unavailable