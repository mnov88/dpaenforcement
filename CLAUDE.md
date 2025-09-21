# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for gathering and analyzing GDPR enforcement data from Data Protection Authorities (DPAs) for comparative and statistical analysis. Currently focused on Norwegian DPA decisions, the project aims to provide insights into regulatory patterns, compliance trends, and enforcement effectiveness across the European data protection landscape.

## Data Structure

The main dataset is stored in `dataNorway.csv` with 50 columns (A1-A50) containing structured GDPR enforcement case data:

- **Jurisdictional data**: Country, authority, decision dates, cross-border status
- **Defendant information**: Names, roles, economic sectors, institutional identity
- **Violation details**: GDPR articles evaluated/violated, legal basis, data types
- **Enforcement outcomes**: Sanctions, fine amounts, compliance measures
- **Case characteristics**: Data subjects affected, infringement duration, damage assessment

Each row represents a single DPA decision with standardized categorical values and free-text fields for case summaries.

## Analysis Framework

The project follows a descriptive statistics methodology focused on:

1. **Geographic enforcement patterns** - Variations across DPAs and jurisdictions
2. **Temporal trends** - Evolution of enforcement over time
3. **Sectoral compliance profiles** - Industry-specific violation patterns
4. **Legal article analysis** - Most violated GDPR provisions and co-occurrence patterns
5. **Fine determinant analysis** - Factors influencing penalty amounts

## Development Environment

This is a data analysis project using Python ecosystem tools:

### Core Libraries
- **pandas** - Data manipulation and CSV processing
- **numpy** - Numerical computations and statistical measures
- **matplotlib & seaborn** - Statistical visualization and publication-ready charts
- **scipy.stats** - Statistical tests and hypothesis testing
- **plotly** - Interactive visualizations for multi-dimensional analysis

### Data Processing
The main data file (`dataNorway.csv`) contains 50 structured columns following a detailed questionnaire format. Each field has specific allowed values documented in the README.md file.

## Key Files

- `README.md` - Complete data structure documentation and field definitions
- `Phase1Plan.md` - Detailed methodology and implementation examples for descriptive analysis
- `dataNorway.csv` - Main dataset with Norwegian GDPR enforcement cases

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