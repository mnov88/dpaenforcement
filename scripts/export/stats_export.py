from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

try:
    import pyreadstat
    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False

from .base import BaseExporter


class StatisticalPackageExporter(BaseExporter):
    """Export data for statistical analysis packages (R, Stata, SPSS)."""

    def __init__(self, wide_csv: Path, long_tables_dir: Optional[Path] = None):
        super().__init__(wide_csv, long_tables_dir)

    def export(self, output_dir: Path, formats: Optional[List[str]] = None) -> None:
        """Export to statistical package formats."""
        if formats is None:
            formats = ['r', 'stata', 'spss']

        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data with statistical semantics
        df_stats = self._prepare_for_stats()

        # Export to requested formats
        if 'r' in formats:
            self._export_r_format(df_stats, output_dir)

        if 'stata' in formats:
            self._export_stata_format(df_stats, output_dir)

        if 'spss' in formats:
            self._export_spss_format(df_stats, output_dir)

        # Export metadata and codebook
        self._export_statistical_metadata(output_dir)
        self._export_codebook(df_stats, output_dir)

    def _prepare_for_stats(self) -> pd.DataFrame:
        """Prepare DataFrame with proper statistical typing and labels."""
        df = self.df.copy()

        # Convert categorical variables with proper ordering
        categorical_mappings = {
            'country_group': {
                'categories': ['EU', 'EEA', 'NON_EEA'],
                'ordered': False,
                'label': 'Geographic grouping of deciding authority'
            },
            'decision_quarter': {
                'categories': ['Q1', 'Q2', 'Q3', 'Q4'],
                'ordered': True,
                'label': 'Quarter of decision year'
            },
            'decision_date_status': {
                'categories': ['DISCUSSED', 'NOT_DISCUSSED'],
                'ordered': False,
                'label': 'Whether decision date was discussed'
            },
            'fine_status': {
                'categories': ['DISCUSSED', 'NOT_MENTIONED', 'PARSE_ERROR'],
                'ordered': False,
                'label': 'Status of fine amount information'
            }
        }

        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = pd.Categorical(
                    df[col],
                    categories=mapping['categories'],
                    ordered=mapping['ordered']
                )

        # Create factor-level encodings for statistical software
        df = self._create_factor_encodings(df)

        # Handle missing values appropriately for statistical analysis
        df = self._handle_statistical_missingness(df)

        # Create derived variables for analysis
        df = self._create_analysis_variables(df)

        return df

    def _create_factor_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create integer factor encodings with labels."""
        # Country codes to numeric for easier analysis
        if 'country_code' in df.columns:
            country_mapping = {code: i+1 for i, code in enumerate(sorted(df['country_code'].dropna().unique()))}
            df['country_code_numeric'] = df['country_code'].map(country_mapping)

        # DPA names to numeric
        if 'dpa_name_canonical' in df.columns:
            dpa_mapping = {dpa: i+1 for i, dpa in enumerate(sorted(df['dpa_name_canonical'].dropna().unique()))}
            df['dpa_numeric'] = df['dpa_name_canonical'].map(dpa_mapping)

        # ISIC sections to numeric
        if 'isic_section' in df.columns:
            isic_mapping = {section: i+1 for i, section in enumerate(sorted(df['isic_section'].dropna().unique()))}
            df['isic_section_numeric'] = df['isic_section'].map(isic_mapping)

        return df

    def _handle_statistical_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values for statistical analysis."""
        # Convert empty strings to NaN for numeric variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create missingness indicators for key variables
        key_vars = ['fine_eur', 'turnover_eur', 'decision_date']
        for var in key_vars:
            if var in df.columns:
                df[f'{var}_missing'] = df[var].isna().astype(int)

        return df

    def _create_analysis_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create variables commonly used in statistical analysis."""
        # Time variables
        if 'decision_date' in df.columns:
            df['decision_date_parsed'] = pd.to_datetime(df['decision_date'], errors='coerce', utc=True)
            gdpr_date = pd.Timestamp('2018-05-25', tz='UTC')
            df['days_since_gdpr'] = (df['decision_date_parsed'] - gdpr_date).dt.days

        # Fine analysis variables
        if 'fine_eur' in df.columns:
            df['fine_positive_flag'] = (df['fine_eur'] > 0).astype(int)
            df['fine_log'] = np.log(df['fine_eur'].replace(0, np.nan))  # Natural log excluding zeros
            df['fine_sqrt'] = np.sqrt(df['fine_eur'])

            # Fine categories for analysis
            conditions = [
                (df['fine_eur'] == 0),
                (df['fine_eur'] > 0) & (df['fine_eur'] <= 10000),
                (df['fine_eur'] > 10000) & (df['fine_eur'] <= 100000),
                (df['fine_eur'] > 100000) & (df['fine_eur'] <= 1000000),
                (df['fine_eur'] > 1000000)
            ]
            choices = [0, 1, 2, 3, 4]  # 0=No fine, 1=Low, 2=Medium, 3=High, 4=Very high
            df['fine_category'] = np.select(conditions, choices, default=-1)

        # Violation intensity measures
        violation_vars = ['n_principles_discussed', 'n_principles_violated', 'n_corrective_measures']
        for var in violation_vars:
            if var in df.columns:
                df[f'{var}_high'] = (df[var] > df[var].median()).astype(int)

        # Enforcement severity index
        severity_components = ['fine_positive_flag', 'severity_measures_present']
        if all(col in df.columns for col in severity_components):
            df['enforcement_severity_index'] = df[severity_components].sum(axis=1)

        return df

    def _export_r_format(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Export R-compatible formats."""
        # Export as RDS (R Data Serialization)
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr

            pandas2ri.activate()
            base = importr('base')

            # Convert to R data frame with proper factor levels
            r_df = self._convert_to_r_dataframe(df)

            # Save as RDS
            base.saveRDS(r_df, str(output_dir / "gdpr_decisions.rds"))

        except ImportError:
            # Fallback: export as CSV with R script for proper loading
            df.to_csv(output_dir / "gdpr_decisions.csv", index=False)
            self._create_r_loading_script(df, output_dir)

        # Create R analysis template
        self._create_r_analysis_template(output_dir)

    def _convert_to_r_dataframe(self, df: pd.DataFrame):
        """Convert pandas DataFrame to R data frame with proper types."""
        # This would require rpy2, implementing fallback for now
        return df

    def _create_r_loading_script(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create R script to properly load the CSV data."""
        variable_info = self._get_variable_info(df)

        r_script = f'''
# GDPR DPA Decisions Dataset Loading Script
# Generated automatically from Python export

library(dplyr)
library(readr)
library(forcats)

# Load the data
gdpr_data <- read_csv("gdpr_decisions.csv")

# Convert categorical variables to factors with proper levels
'''

        # Add factor conversions
        for var, info in variable_info.items():
            if info.get('type') == 'categorical':
                levels = info.get('levels', [])
                if levels:
                    levels_str = ', '.join(f'"{level}"' for level in levels)
                    ordered = 'TRUE' if info.get('ordered', False) else 'FALSE'
                    r_script += f'''gdpr_data${var} <- factor(gdpr_data${var}, levels = c({levels_str}), ordered = {ordered})\n'''

        r_script += '''
# Convert date variables
gdpr_data$decision_date <- as.Date(gdpr_data$decision_date)

# Add variable labels (requires Hmisc package)
if (require(Hmisc)) {
'''

        # Add variable labels
        for var, info in variable_info.items():
            label = info.get('label', var)
            r_script += f'  label(gdpr_data${var}) <- "{label}"\n'

        r_script += '''
}

# Summary statistics
cat("Dataset loaded successfully\\n")
cat("Dimensions:", nrow(gdpr_data), "rows,", ncol(gdpr_data), "columns\\n")
cat("Decision date range:", min(gdpr_data$decision_date, na.rm=TRUE), "to", max(gdpr_data$decision_date, na.rm=TRUE), "\\n")

# Basic descriptive statistics
summary(gdpr_data[c("fine_eur", "decision_year", "country_group", "breach_case")])
'''

        with open(output_dir / "load_gdpr_data.R", "w") as f:
            f.write(r_script)

    def _create_r_analysis_template(self, output_dir: Path) -> None:
        """Create R analysis template with common GDPR research patterns."""
        template = '''
# GDPR DPA Decisions Analysis Template
# Legal data analysis with proper handling of typed missingness

library(dplyr)
library(ggplot2)
library(forcats)
library(lubridate)

# Load the data (run load_gdpr_data.R first)
source("load_gdpr_data.R")

# ============================================================================
# 1. DESCRIPTIVE ANALYSIS
# ============================================================================

# Fine distribution by country group
gdpr_data %>%
  filter(fine_eur > 0) %>%
  ggplot(aes(x = country_group, y = fine_eur)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "Fine Distribution by Geographic Region",
       y = "Fine Amount (EUR, log scale)",
       x = "Geographic Group")

# Temporal trends in enforcement
temporal_trends <- gdpr_data %>%
  filter(!is.na(decision_year)) %>%
  group_by(decision_year, country_group) %>%
  summarise(
    n_decisions = n(),
    n_fines = sum(fine_positive_flag, na.rm = TRUE),
    median_fine = median(fine_eur[fine_eur > 0], na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================================
# 2. VIOLATION ANALYSIS (Respecting Legal Semantics)
# ============================================================================

# Article 5 violations (only where discussed)
article5_analysis <- gdpr_data %>%
  filter(q31_violated_status == "DISCUSSED") %>%  # Only analyze where violations were assessed
  select(decision_id, starts_with("q31_violated_")) %>%
  select(-ends_with("_status"), -ends_with("_coverage_status")) %>%
  pivot_longer(cols = -c(decision_id), names_to = "principle", values_to = "violated") %>%
  filter(!is.na(violated) & violated == 1) %>%
  count(principle, sort = TRUE)

# ============================================================================
# 3. ENFORCEMENT PATTERNS
# ============================================================================

# Fine determinants (basic model - extend as needed)
fine_model <- gdpr_data %>%
  filter(fine_status == "DISCUSSED") %>%  # Only where fines were assessed
  lm(fine_log ~ country_group + breach_case + n_principles_violated +
     enforcement_severity_index, data = .)

summary(fine_model)

# ============================================================================
# 4. LEGAL INTERPRETATION NOTES
# ============================================================================

# IMPORTANT: This dataset preserves legal nuance through typed missingness
# - NOT_DISCUSSED != NOT_VIOLATED
# - Always filter by status variables before analysis
# - Use appropriate statistical methods for legal data

cat("Analysis template completed. Modify as needed for specific research questions.\\n")
'''

        with open(output_dir / "analysis_template.R", "w") as f:
            f.write(template)

    def _export_stata_format(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Export Stata-compatible format."""
        if not PYREADSTAT_AVAILABLE:
            # Fallback to CSV with Stata do-file
            df.to_csv(output_dir / "gdpr_decisions.csv", index=False)
            self._create_stata_loading_script(df, output_dir)
            return

        # Prepare data for Stata
        df_stata = df.copy()

        # Create variable labels
        variable_labels = self._get_stata_variable_labels(df_stata)

        # Create value labels for categorical variables
        value_labels = self._get_stata_value_labels(df_stata)

        try:
            # Export to Stata format
            pyreadstat.write_dta(
                df_stata,
                str(output_dir / "gdpr_decisions.dta"),
                variable_labels=variable_labels,
                value_labels=value_labels
            )
        except Exception:
            # Fallback to CSV
            df.to_csv(output_dir / "gdpr_decisions.csv", index=False)
            self._create_stata_loading_script(df, output_dir)

    def _create_stata_loading_script(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create Stata do-file for proper data loading."""
        variable_info = self._get_variable_info(df)

        do_file = '''* GDPR DPA Decisions Dataset Loading Script
* Generated automatically from Python export

clear all
set more off

* Import CSV data
import delimited "gdpr_decisions.csv", clear

* Variable labels and formatting
'''

        # Add variable labels and formats
        for var, info in variable_info.items():
            label = info.get('label', var)
            do_file += f'label var {var} "{label}"\n'

            # Add formats for specific variable types
            if var.endswith('_eur'):
                do_file += f'format {var} %12.2fc\n'
            elif var in ['decision_date']:
                do_file += f'generate {var}_stata = date({var}, "YMD")\n'
                do_file += f'format {var}_stata %td\n'

        # Add value labels for categorical variables
        do_file += '''
* Value labels for categorical variables
label define country_group 1 "EU" 2 "EEA" 3 "NON_EEA"
label define quarter 1 "Q1" 2 "Q2" 3 "Q3" 4 "Q4"
label define yesno 0 "No" 1 "Yes"

* Apply value labels where appropriate
'''

        # Add encoding instructions
        categorical_vars = [var for var, info in variable_info.items()
                          if info.get('type') == 'categorical']

        for var in categorical_vars:
            if 'group' in var:
                do_file += f'encode {var}, gen({var}_coded) label(country_group)\n'

        do_file += '''
* Summary statistics
describe
summarize fine_eur decision_year, detail

* Dataset information
display "Dataset loaded successfully"
display "Observations: " _N
display "Variables: " r(k)

* Save as Stata dataset
save "gdpr_decisions_final.dta", replace
'''

        with open(output_dir / "load_gdpr_data.do", "w") as f:
            f.write(do_file)

    def _get_stata_variable_labels(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get variable labels for Stata export."""
        variable_info = self._get_variable_info(df)
        return {var: info.get('label', var) for var, info in variable_info.items()}

    def _get_stata_value_labels(self, df: pd.DataFrame) -> Dict[str, Dict[Any, str]]:
        """Get value labels for Stata categorical variables."""
        value_labels = {}

        # Country group labels
        if 'country_group' in df.columns:
            value_labels['country_group'] = {
                'EU': 'European Union',
                'EEA': 'European Economic Area',
                'NON_EEA': 'Non-EEA Country'
            }

        # Quarter labels
        if 'decision_quarter' in df.columns:
            value_labels['decision_quarter'] = {
                'Q1': 'First Quarter',
                'Q2': 'Second Quarter',
                'Q3': 'Third Quarter',
                'Q4': 'Fourth Quarter'
            }

        return value_labels

    def _export_spss_format(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Export SPSS-compatible format."""
        if not PYREADSTAT_AVAILABLE:
            df.to_csv(output_dir / "gdpr_decisions.csv", index=False)
            return

        # Prepare for SPSS
        df_spss = df.copy()

        # SPSS variable labels
        variable_labels = self._get_spss_variable_labels(df_spss)

        # SPSS value labels
        value_labels = self._get_spss_value_labels(df_spss)

        try:
            pyreadstat.write_sav(
                df_spss,
                str(output_dir / "gdpr_decisions.sav"),
                variable_labels=variable_labels,
                value_labels=value_labels
            )
        except Exception:
            df.to_csv(output_dir / "gdpr_decisions.csv", index=False)

    def _get_spss_variable_labels(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get variable labels for SPSS export."""
        return self._get_stata_variable_labels(df)  # Same format

    def _get_spss_value_labels(self, df: pd.DataFrame) -> Dict[str, Dict[Any, str]]:
        """Get value labels for SPSS categorical variables."""
        return self._get_stata_value_labels(df)  # Same format

    def _get_variable_info(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive variable information for all formats."""
        info = {}

        # Define variable metadata
        base_vars = {
            'decision_id': {'label': 'Unique decision identifier', 'type': 'string'},
            'country_code': {'label': 'ISO country code of deciding authority', 'type': 'categorical'},
            'country_group': {'label': 'Geographic grouping (EU/EEA/NON_EEA)', 'type': 'categorical',
                            'levels': ['EU', 'EEA', 'NON_EEA']},
            'dpa_name_canonical': {'label': 'Standardized DPA name', 'type': 'string'},
            'decision_date': {'label': 'Date of decision issuance', 'type': 'date'},
            'decision_year': {'label': 'Year of decision', 'type': 'numeric'},
            'decision_quarter': {'label': 'Quarter of decision year', 'type': 'categorical',
                               'levels': ['Q1', 'Q2', 'Q3', 'Q4'], 'ordered': True},
            'fine_eur': {'label': 'Administrative fine amount (EUR)', 'type': 'numeric'},
            'turnover_eur': {'label': 'Annual turnover mentioned (EUR)', 'type': 'numeric'},
            'fine_positive': {'label': 'Whether fine > 0 was imposed', 'type': 'binary'},
            'breach_case': {'label': 'Whether data breach was discussed', 'type': 'binary'},
            'severity_measures_present': {'label': 'Whether severe measures imposed', 'type': 'binary'},
            'n_principles_discussed': {'label': 'Number of Article 5 principles discussed', 'type': 'count'},
            'n_principles_violated': {'label': 'Number of Article 5 principles violated', 'type': 'count'},
            'n_corrective_measures': {'label': 'Number of corrective measures imposed', 'type': 'count'}
        }

        # Add information for variables present in the dataset
        for var in df.columns:
            if var in base_vars:
                info[var] = base_vars[var]
            elif var.startswith('q30_discussed_'):
                principle = var.replace('q30_discussed_', '')
                info[var] = {'label': f'Article 5 principle discussed: {principle}', 'type': 'binary'}
            elif var.startswith('q31_violated_'):
                principle = var.replace('q31_violated_', '')
                info[var] = {'label': f'Article 5 principle violated: {principle}', 'type': 'binary'}
            elif var.startswith('q53_powers_'):
                power = var.replace('q53_powers_', '')
                info[var] = {'label': f'Corrective power exercised: {power}', 'type': 'binary'}
            elif var.endswith('_status'):
                base_var = var.replace('_status', '')
                info[var] = {'label': f'Status of {base_var} information', 'type': 'categorical'}
            else:
                info[var] = {'label': var.replace('_', ' ').title(), 'type': 'unknown'}

        return info

    def _export_statistical_metadata(self, output_dir: Path) -> None:
        """Export metadata specific to statistical analysis."""
        metadata = {
            'statistical_formats': {
                'r_rds': 'R data serialization format with factor levels preserved',
                'stata_dta': 'Stata dataset with variable and value labels',
                'spss_sav': 'SPSS dataset with comprehensive metadata'
            },
            'legal_data_considerations': {
                'typed_missingness': 'NOT_DISCUSSED != NOT_MENTIONED != NOT_APPLICABLE',
                'violation_analysis': 'Filter by status variables before analyzing violations',
                'fine_analysis': 'Zero fines may be meaningful (no fine imposed)',
                'temporal_analysis': 'Account for GDPR implementation date (2018-05-25)'
            },
            'variable_categories': {
                'identifiers': ['decision_id'],
                'geographic': ['country_code', 'country_group'],
                'temporal': ['decision_date', 'decision_year', 'decision_quarter'],
                'monetary': ['fine_eur', 'turnover_eur', 'fine_log', 'fine_sqrt'],
                'legal_outcomes': ['breach_case', 'severity_measures_present', 'remedy_only_case'],
                'violation_indicators': 'Variables starting with q30_, q31_, q53_, q56_, q57_',
                'status_variables': 'Variables ending with _status'
            },
            'analysis_recommendations': {
                'regression_analysis': [
                    'Use robust standard errors for fine analysis',
                    'Consider log transformation for monetary variables',
                    'Account for country-level clustering',
                    'Filter by status variables to avoid bias'
                ],
                'categorical_analysis': [
                    'Use chi-square tests with Bonferroni correction',
                    'Account for legal exclusivity in multi-select variables',
                    'Consider ordinal nature of severity measures'
                ],
                'time_series_analysis': [
                    'Account for GDPR learning curve effects',
                    'Consider seasonal patterns in enforcement',
                    'Use appropriate trend tests for legal data'
                ]
            }
        }

        with open(output_dir / "statistical_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _export_codebook(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Export comprehensive codebook for the dataset."""
        variable_info = self._get_variable_info(df)

        codebook = {
            'dataset_info': {
                'title': 'GDPR DPA Decisions Analysis Dataset',
                'description': 'Structured dataset of GDPR enforcement decisions with 68-field questionnaire responses',
                'observations': len(df),
                'variables': len(df.columns),
                'legal_framework': 'EU GDPR (Regulation 2016/679)',
                'data_collection_period': f"{df['decision_year'].min()} - {df['decision_year'].max()}" if 'decision_year' in df.columns else 'Unknown'
            },
            'variables': {}
        }

        # Add detailed variable information
        for var in df.columns:
            var_info = variable_info.get(var, {})

            var_details = {
                'label': var_info.get('label', var),
                'type': var_info.get('type', 'unknown'),
                'missing_count': int(df[var].isna().sum()),
                'missing_rate': float(df[var].isna().mean())
            }

            # Add type-specific statistics
            if var_info.get('type') == 'numeric':
                var_details.update({
                    'min': float(df[var].min()) if df[var].dtype in ['int64', 'float64'] else None,
                    'max': float(df[var].max()) if df[var].dtype in ['int64', 'float64'] else None,
                    'mean': float(df[var].mean()) if df[var].dtype in ['int64', 'float64'] else None,
                    'std': float(df[var].std()) if df[var].dtype in ['int64', 'float64'] else None
                })
            elif var_info.get('type') in ['categorical', 'binary']:
                value_counts = df[var].value_counts().to_dict()
                var_details['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
                if var_info.get('levels'):
                    var_details['levels'] = var_info['levels']

            codebook['variables'][var] = var_details

        # Export as JSON
        with open(output_dir / "codebook.json", "w", encoding="utf-8") as f:
            json.dump(codebook, f, indent=2, ensure_ascii=False, default=str)

        # Export as human-readable text
        self._export_text_codebook(codebook, output_dir)

    def _export_text_codebook(self, codebook: Dict[str, Any], output_dir: Path) -> None:
        """Export human-readable text codebook."""
        text_content = f"""
GDPR DPA DECISIONS ANALYSIS DATASET - CODEBOOK
===============================================

Dataset Information:
- Title: {codebook['dataset_info']['title']}
- Observations: {codebook['dataset_info']['observations']:,}
- Variables: {codebook['dataset_info']['variables']:,}
- Legal Framework: {codebook['dataset_info']['legal_framework']}
- Data Period: {codebook['dataset_info'].get('data_collection_period', 'Unknown')}

VARIABLE DESCRIPTIONS:
======================

"""

        # Group variables by category for better organization
        categories = {
            'Identifiers': [],
            'Geographic Variables': [],
            'Temporal Variables': [],
            'Monetary Variables': [],
            'Legal Outcomes': [],
            'Article 5 Principles': [],
            'Data Subject Rights': [],
            'Corrective Powers': [],
            'Status Variables': [],
            'Other Variables': []
        }

        for var, info in codebook['variables'].items():
            if var in ['decision_id']:
                categories['Identifiers'].append((var, info))
            elif any(x in var for x in ['country', 'dpa']):
                categories['Geographic Variables'].append((var, info))
            elif any(x in var for x in ['date', 'year', 'quarter']):
                categories['Temporal Variables'].append((var, info))
            elif any(x in var for x in ['fine', 'turnover', 'eur']):
                categories['Monetary Variables'].append((var, info))
            elif var.startswith('q30_') or var.startswith('q31_'):
                categories['Article 5 Principles'].append((var, info))
            elif var.startswith('q56_') or var.startswith('q57_'):
                categories['Data Subject Rights'].append((var, info))
            elif var.startswith('q53_'):
                categories['Corrective Powers'].append((var, info))
            elif var.endswith('_status'):
                categories['Status Variables'].append((var, info))
            elif any(x in var for x in ['breach', 'severity', 'remedy']):
                categories['Legal Outcomes'].append((var, info))
            else:
                categories['Other Variables'].append((var, info))

        for category, variables in categories.items():
            if variables:
                text_content += f"\n{category}:\n" + "-" * len(category) + "\n"
                for var, info in variables:
                    text_content += f"\n{var}\n"
                    text_content += f"  Label: {info['label']}\n"
                    text_content += f"  Type: {info['type']}\n"
                    text_content += f"  Missing: {info['missing_count']} ({info['missing_rate']:.1%})\n"

                    if 'value_counts' in info:
                        text_content += "  Values:\n"
                        for value, count in sorted(info['value_counts'].items()):
                            text_content += f"    {value}: {count}\n"

                    if 'min' in info and info['min'] is not None:
                        text_content += f"  Range: {info['min']} - {info['max']}\n"
                        text_content += f"  Mean: {info['mean']:.2f}, Std: {info['std']:.2f}\n"

        text_content += """

IMPORTANT NOTES FOR STATISTICAL ANALYSIS:
==========================================

1. Legal Data Semantics:
   - NOT_DISCUSSED ≠ NOT_MENTIONED ≠ NOT_APPLICABLE
   - Always filter by status variables before analyzing violations
   - Zero fines may be meaningful (explicit decision of no fine)

2. Multi-select Variables:
   - Binary indicators represent presence/absence of each option
   - Check exclusivity_conflict flags for contradictory responses
   - Use coverage_status to understand response completeness

3. Missing Data:
   - Status variables preserve legal interpretation of missingness
   - Use appropriate imputation methods that respect legal semantics
   - Consider legal reasons for non-response

4. Temporal Analysis:
   - Account for GDPR implementation date (2018-05-25)
   - Consider learning curve effects in early decisions
   - Seasonal patterns may reflect administrative cycles

5. Cross-Country Analysis:
   - Control for different legal traditions and enforcement approaches
   - Consider clustering of standard errors by country/DPA
   - Account for varying interpretation of GDPR requirements
"""

        with open(output_dir / "codebook.txt", "w", encoding="utf-8") as f:
            f.write(text_content)