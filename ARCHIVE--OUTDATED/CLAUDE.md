# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for gathering and analyzing GDPR enforcement data from Data Protection Authorities (DPAs) for comparative and statistical analysis. Currently covering Norwegian and Swedish DPA decisions with support for combined Nordic analysis, the project aims to provide insights into regulatory patterns, compliance trends, and enforcement effectiveness across the European data protection landscape.

## Data Structure

The datasets are stored in country-specific CSV files (`dataNorway.csv`, `dataSweden.csv`) with 50 base columns (A1-A50) containing structured GDPR enforcement case data. Enhanced processing creates combined datasets with 200+ analytical columns:

- **Jurisdictional data**: Country, authority, decision dates, cross-border status
- **Defendant information**: Names, roles, economic sectors, institutional identity
- **Violation details**: GDPR articles evaluated/violated, legal basis, data types
- **Enforcement outcomes**: Sanctions, fine amounts, compliance measures
- **Case characteristics**: Data subjects affected, infringement duration, damage assessment

Each row represents a single DPA decision with standardized categorical values and free-text fields for case summaries.

## Multi-Country Analysis Framework

The project supports both single-country and comparative multi-country analysis:

### Supported Countries
- **Norway**: Comprehensive dataset of Norwegian DPA (Datatilsynet) enforcement decisions
- **Sweden**: Swedish DPA enforcement decisions with identical data structure
- **Combined Nordic**: Merged dataset enabling cross-country comparative analysis

### Cross-Country Capabilities
- Unified data schema across all countries (50 base fields)
- Standardized cleaning and enhancement pipeline
- Currency normalization to EUR for fair comparison
- Combined analytical features for comparative studies
- Provenance tracking to identify data source country

## Analysis Framework

The project follows a descriptive statistics methodology focused on:

1. **Geographic enforcement patterns** - Variations across DPAs and jurisdictions
2. **Temporal trends** - Evolution of enforcement over time
3. **Sectoral compliance profiles** - Industry-specific violation patterns
4. **Legal article analysis** - Most violated GDPR provisions and co-occurrence patterns
5. **Fine determinant analysis** - Factors influencing penalty amounts

## Enhanced Data Processing

The project includes advanced data processing capabilities that significantly expand analytical possibilities:

### JSON Reporting System
- Comprehensive timestamped reports for all processing stages
- `data_cleaning_report_*.json` - Detailed logs of cleaning operations
- `schema_validation_report_*.json` - Validation results and error tracking
- `univariate_analysis_results_*.json` - Statistical analysis outputs
- `bivariate_analysis_results_*.json` - Relationship analysis results
- Complete audit trail with metadata and processing statistics

### 194 Derived Analytical Columns
The enhanced cleaning pipeline expands the original 50 columns to 200+ features:

**One-Hot Encoded Categories** (37 columns):
- Article indicators: `A34_Art_*`, `A35_Art_*` (top-level GDPR articles)
- Enum expansions: A4, A6, A8, A9, A12, A15, A27-A30, A33, A37-A44, A49

**Binary Indicators** (15 columns):
- Y/N conversions: `{field}_bin` for A5, A7, A16, A18, A21, A31, A36, A48
- Tri-state positive: `{field}_pos_bin` for A23, A24, A27-A30, A33, A37-A43

**Multi-Select Expansions** (38 columns):
- `SensitiveType_*` (8 types), `VulnerableSubject_*` (7 types)
- `LegalBasis_*` (6 types), `TransferMech_*` (6 types)
- `Right_*` (7 types), `Sanction_*` (6 types)

**Parsed Structured Fields** (8 columns):
- `A13_SNA_Code`, `A13_SNA_Desc` (institutional identity)
- `A14_ISIC_Code`, `A14_ISIC_Desc`, `A14_ISIC_Level` (economic sector)
- `A25_SubjectsAffected_*` (min/max/midpoint/is_range)

**Enhanced Features** (6 columns):
- `A26_Duration_Months` (standardized time periods)
- `A46_FineAmount_EUR` (currency-normalized fines)
- `dataset_source` (provenance tracking)

## Development Environment

This is a data analysis project using Python ecosystem tools:

### Core Libraries
- **pandas** - Data manipulation and CSV processing
- **numpy** - Numerical computations and statistical measures
- **matplotlib & seaborn** - Statistical visualization and publication-ready charts
- **scipy.stats** - Statistical tests and hypothesis testing
- **plotly** - Interactive visualizations for multi-dimensional analysis

### Data Processing
The base data files (`dataNorway.csv`, `dataSweden.csv`) contain 50 structured columns following a detailed questionnaire format. After enhanced processing, cleaned datasets expand to 200+ columns including 194 derived analytical features (one-hot encoded categoricals, parsed structured fields, currency normalization, temporal features, and binary indicators). Each field has specific allowed values documented in the README.md file.

## Key Files

- `README.md` - Complete data structure documentation and field definitions
- `Phase1Plan.md` - Detailed methodology and implementation examples for descriptive analysis
- `PREPROCESSING_DOCUMENTATION.md` - Comprehensive enhanced data cleaning and processing documentation
- `dataNorway.csv` - Norwegian GDPR enforcement cases dataset
- `dataSweden.csv` - Swedish GDPR enforcement cases dataset
- `dataNordics_cleaned_combined.csv` - Combined and cleaned Nordic dataset for cross-country analysis
- `enhanced_data_cleaning.py` - Advanced data preprocessing pipeline with 194 derived analytical features
- `multi_format_exporter.py` - Export system supporting CSV, Excel, Parquet, and JSON formats with metadata

## Analysis Best Practices

When working with this data:

1. **Handle missing values systematically** - Document imputation strategies
2. **Normalize financial data** - Convert fines to common currency (EUR) with historical rates
3. **Parse multi-select fields** - Many columns contain comma-separated values requiring splitting
4. **Extract GDPR articles** - Use regex to parse article references from free-text fields
5. **Account for temporal effects** - Early GDPR enforcement may differ from current patterns
6. **Consider jurisdictional differences** - Legal systems and reporting standards vary

## Data Quality Considerations

- Some fields may have inconsistent formatting (especially dates and article references)
- Missing data patterns may be informative rather than random
- Cross-border cases have additional complexity in role assignments
- Fine amounts need currency conversion and inflation adjustment for temporal analysis