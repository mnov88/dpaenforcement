from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class BaseExporter(ABC):
    """Base class for all data exporters with common functionality."""

    def __init__(self, wide_csv: Path, long_tables_dir: Optional[Path] = None):
        self.wide_csv = wide_csv
        self.long_tables_dir = long_tables_dir
        self._df_cache: Optional[pd.DataFrame] = None
        self._long_cache: Dict[str, pd.DataFrame] = {}

    @property
    def df(self) -> pd.DataFrame:
        """Cached wide DataFrame with proper typing."""
        if self._df_cache is None:
            self._df_cache = self._load_wide_data()
        return self._df_cache

    def _load_wide_data(self) -> pd.DataFrame:
        """Load and type-cast wide CSV data."""
        df = pd.read_csv(self.wide_csv)

        # Convert boolean indicators
        bool_cols = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in [
                'q30_discussed_', 'q31_violated_', 'q32_bases_', 'q41_aggrav_',
                'q42_mitig_', 'q43_harm_', 'q44_benefit_', 'q45_coop_',
                'q46_vuln_', 'q47_remedial_', 'q50_other_measures_',
                'q53_powers_', 'q54_scopes_', 'q56_rights_discussed_', 'q57_rights_violated_',
                'q58_access_issues_', 'q59_adm_issues_', 'q61_dpo_issues_', 'q64_transfer_violations_'
            ]
        ) and not any(col.endswith(suffix) for suffix in [
            '_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'
        ])]

        for col in bool_cols:
            df[col] = df[col].replace({'': None}).astype('boolean')

        # Convert numeric columns
        numeric_cols = ['fine_eur', 'turnover_eur', 'fine_log1p', 'turnover_log1p',
                       'fine_to_turnover_ratio', 'decision_year']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert dates
        if 'decision_date' in df.columns:
            df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

        # Convert categorical fields with legal semantics
        categorical_mappings = {
            'country_group': ['EU', 'EEA', 'NON_EEA'],
            'decision_date_status': ['DISCUSSED', 'NOT_DISCUSSED'],
            'fine_status': ['DISCUSSED', 'NOT_MENTIONED', 'PARSE_ERROR'],
            'turnover_status': ['DISCUSSED', 'NOT_MENTIONED', 'PARSE_ERROR']
        }

        for col, categories in categorical_mappings.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=categories)

        return df

    def load_long_table(self, table_name: str) -> pd.DataFrame:
        """Load and cache a specific long table."""
        if table_name not in self._long_cache:
            if not self.long_tables_dir:
                raise ValueError("Long tables directory not provided")

            table_path = self.long_tables_dir / table_name
            if not table_path.exists():
                raise FileNotFoundError(f"Long table not found: {table_path}")

            df = pd.read_csv(table_path)
            df['token_status'] = pd.Categorical(df['token_status'],
                                              categories=['KNOWN', 'UNKNOWN', 'STATUS_ONLY'])
            self._long_cache[table_name] = df

        return self._long_cache[table_name]

    def get_legal_metadata(self) -> Dict[str, Any]:
        """Return metadata about legal field semantics."""
        return {
            'typed_missingness': {
                'NOT_DISCUSSED': 'Question not addressed in decision',
                'NOT_MENTIONED': 'Aspect not mentioned in response',
                'NOT_APPLICABLE': 'Question not relevant to case type',
                'NOT_DETERMINED': 'Violation status unclear from decision',
                'UNCLEAR': 'Response ambiguous or incomplete'
            },
            'multi_select_exclusivity': {
                'NONE_DISCUSSED': 'Excludes substantive options',
                'NONE_VIOLATED': 'Explicit finding of no violation',
                'NONE_MENTIONED': 'No relevant options present'
            },
            'question_mapping': {
                'Q30': 'Article 5 principles discussed',
                'Q31': 'Article 5 principles violated',
                'Q32': 'Article 6 legal bases discussed',
                'Q33': 'Legal bases relied upon by defendant',
                'Q41': 'Article 83(2) aggravating factors',
                'Q42': 'Article 83(2) mitigating factors',
                'Q43': 'Documented harm to data subjects',
                'Q44': 'Economic benefit to defendant',
                'Q45': 'Defendant cooperation level',
                'Q53': 'Article 58(2) corrective powers exercised',
                'Q56': 'Data subject rights discussed',
                'Q57': 'Data subject rights violated'
            }
        }

    @abstractmethod
    def export(self, output_path: Path) -> None:
        """Export data in the specific format."""
        pass


class SchemaRegistry:
    """Centralized schema definitions for legal data."""

    @staticmethod
    def get_wide_schema() -> Dict[str, str]:
        """Return pandas-compatible dtypes for wide CSV."""
        return {
            'decision_id': 'string',
            'country_code': 'category',
            'country_group': 'category',
            'dpa_name_canonical': 'string',
            'decision_date': 'datetime64[ns]',
            'decision_year': 'int64',
            'decision_quarter': 'category',
            'fine_eur': 'float64',
            'turnover_eur': 'float64',
            'fine_positive': 'boolean',
            'fine_log1p': 'float64',
            'turnover_log1p': 'float64',
            'breach_case': 'boolean',
            'severity_measures_present': 'boolean',
            'remedy_only_case': 'boolean',
            'n_principles_discussed': 'int64',
            'n_principles_violated': 'int64',
            'n_corrective_measures': 'int64'
        }

    @staticmethod
    def get_question_descriptions() -> Dict[str, str]:
        """Return human-readable descriptions for questionnaire fields."""
        return {
            'Q1': 'Country of deciding authority',
            'Q2': 'Official name of DPA',
            'Q3': 'Decision issue date',
            'Q16': 'Data breach incident discussed',
            'Q30': 'Article 5 processing principles discussed',
            'Q31': 'Article 5 principles found violated',
            'Q32': 'Article 6 legal bases discussed',
            'Q37': 'Administrative fine amount (EUR)',
            'Q38': 'Annual turnover mentioned (EUR)',
            'Q43': 'Documented harm to data subjects',
            'Q44': 'Economic benefit to defendant',
            'Q45': 'Defendant cooperation level',
            'Q53': 'Article 58(2) corrective powers exercised',
            'Q56': 'Data subject rights discussed',
            'Q57': 'Data subject rights violated'
        }