from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from .base import BaseExporter


class ParquetExporter(BaseExporter):
    """Export data to Parquet format with partitioning and metadata."""

    def __init__(self, wide_csv: Path, long_tables_dir: Optional[Path] = None):
        if not PYARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for Parquet export. Install with: pip install pyarrow"
            )
        super().__init__(wide_csv, long_tables_dir)

    def export(self, output_dir: Path, partition_cols: Optional[List[str]] = None) -> None:
        """Export wide data to partitioned Parquet format."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default partitioning by country group and year for efficient querying
        if partition_cols is None:
            partition_cols = ['country_group', 'decision_year']

        # Prepare DataFrame for export
        df = self.df.copy()

        # Handle missing values appropriately for Parquet
        df = self._prepare_for_parquet(df)

        # Create Arrow table with rich metadata
        table = self._create_arrow_table(df)

        # Write partitioned dataset
        if partition_cols:
            # Ensure partition columns exist and have valid values
            valid_partitions = [col for col in partition_cols if col in df.columns]
            if valid_partitions:
                # Fill NaN in partition columns to avoid errors
                for col in valid_partitions:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('UNKNOWN')
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(-999)

                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_to_dataset(
                    table,
                    root_path=output_dir / "wide_partitioned",
                    partition_cols=valid_partitions,
                    compression='snappy',
                    use_deprecated_int96_timestamps=False
                )
            else:
                # Fallback to non-partitioned if columns don't exist
                table.to_pandas().to_parquet(
                    output_dir / "wide_data.parquet",
                    compression='snappy',
                    index=False
                )
        else:
            # Single file export
            table.to_pandas().to_parquet(
                output_dir / "wide_data.parquet",
                compression='snappy',
                index=False
            )

        # Export long tables if available
        if self.long_tables_dir:
            self._export_long_tables(output_dir)

        # Export metadata
        self._export_metadata(output_dir)

    def _prepare_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Parquet export with proper typing."""
        df = df.copy()

        # Convert empty strings to None for better Parquet compatibility
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            df[col] = df[col].replace('', None)

        # Ensure numeric columns are properly typed
        numeric_mappings = {
            'fine_eur': 'float64',
            'turnover_eur': 'float64',
            'fine_to_turnover_ratio': 'float64',
            'decision_year': 'Int64',  # Nullable integer
            'ingest_line_count': 'Int64',
            'ingest_question_count': 'Int64',
            'ingest_warning_count': 'Int64',
            'q36_tokens': 'Int64',
            'q52_tokens': 'Int64',
            'q67_tokens': 'Int64',
            'q68_tokens': 'Int64'
        }

        for col, dtype in numeric_mappings.items():
            if col in df.columns:
                if dtype.startswith('Int'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert decision quarter to proper categorical
        if 'decision_quarter' in df.columns:
            df['decision_quarter'] = pd.Categorical(
                df['decision_quarter'],
                categories=['Q1', 'Q2', 'Q3', 'Q4'],
                ordered=True
            )

        return df

    def _create_arrow_table(self, df: pd.DataFrame) -> pa.Table:
        """Create Arrow table with rich metadata."""
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Add custom metadata
        metadata = self.get_legal_metadata()
        metadata.update({
            'source': str(self.wide_csv),
            'export_format': 'parquet',
            'schema_version': '1.0',
            'gdpr_questionnaire_version': '68_questions',
            'legal_data_warnings': [
                'Preserve typed missingness semantics',
                'NOT_DISCUSSED != NOT_VIOLATED',
                'Consult validation reports for data quality flags'
            ]
        })

        # Add column descriptions
        column_descriptions = self._get_column_descriptions()

        # Create schema with metadata
        schema_metadata = {
            'dpa_pipeline_metadata': json.dumps(metadata),
            'column_descriptions': json.dumps(column_descriptions)
        }

        new_schema = table.schema.with_metadata(schema_metadata)
        return table.cast(new_schema)

    def _get_column_descriptions(self) -> Dict[str, str]:
        """Generate descriptions for all columns."""
        descriptions = {
            'decision_id': 'Unique identifier for each DPA decision',
            'country_code': 'ISO 3166-1 alpha-2 country code',
            'country_group': 'Geographic grouping (EU/EEA/NON_EEA)',
            'dpa_name_canonical': 'Standardized name of data protection authority',
            'decision_date': 'Date of decision issuance (ISO format)',
            'decision_year': 'Year of decision for temporal analysis',
            'decision_quarter': 'Quarter of decision (Q1-Q4)',
            'fine_eur': 'Administrative fine amount in EUR',
            'turnover_eur': 'Annual turnover mentioned in EUR',
            'fine_positive': 'Boolean: fine amount > 0',
            'fine_log1p': 'Log-transformed fine (log1p for zero-handling)',
            'turnover_log1p': 'Log-transformed turnover',
            'fine_to_turnover_ratio': 'Fine as proportion of turnover',
            'breach_case': 'Boolean: data breach discussed in decision',
            'severity_measures_present': 'Boolean: severe corrective measures imposed',
            'remedy_only_case': 'Boolean: no fine but corrective measures',
            'n_principles_discussed': 'Count of Article 5 principles discussed',
            'n_principles_violated': 'Count of Article 5 principles violated',
            'n_corrective_measures': 'Count of Article 58(2) measures imposed',
            'schema_echo_flag': 'Boolean: AI response contains schema artifacts',
            'isic_code': 'ISIC Rev.4 economic activity code',
            'isic_section': 'ISIC high-level sector classification'
        }

        # Add multi-select column descriptions
        multi_prefixes = {
            'q30_discussed': 'Article 5 principles discussed (binary indicators)',
            'q31_violated': 'Article 5 principles violated (binary indicators)',
            'q32_bases': 'Article 6 legal bases discussed (binary indicators)',
            'q53_powers': 'Article 58(2) corrective powers exercised (binary indicators)',
            'q56_rights_discussed': 'Data subject rights discussed (binary indicators)',
            'q57_rights_violated': 'Data subject rights violated (binary indicators)'
        }

        for prefix, desc in multi_prefixes.items():
            # Add coverage columns
            descriptions[f"{prefix}_coverage_status"] = f"{desc} - overall status"
            descriptions[f"{prefix}_status"] = f"{desc} - derived legal status"
            descriptions[f"{prefix}_exclusivity_conflict"] = f"{desc} - conflicting options flag"

        return descriptions

    def _export_long_tables(self, output_dir: Path) -> None:
        """Export long tables as Parquet files."""
        long_output = output_dir / "long_tables"
        long_output.mkdir(exist_ok=True)

        if not self.long_tables_dir:
            return

        # Key long tables for analysis
        tables = [
            'defendant_classifications.csv', 'breach_types.csv',
            'special_data_categories.csv', 'mitigating_actions.csv',
            'article_5_discussed.csv', 'article_5_violated.csv',
            'article_6_discussed.csv', 'corrective_powers.csv',
            'rights_discussed.csv', 'rights_violated.csv',
            'aggravating_factors.csv', 'mitigating_factors.csv'
        ]

        for table_name in tables:
            try:
                df = self.load_long_table(table_name)
                # Convert to proper dtypes
                df['decision_id'] = df['decision_id'].astype('string')
                df['option'] = df['option'].astype('string')
                df['status'] = pd.Categorical(df['status'])
                df['token_status'] = pd.Categorical(df['token_status'])

                # Export to Parquet
                output_file = long_output / table_name.replace('.csv', '.parquet')
                df.to_parquet(output_file, compression='snappy', index=False)

            except FileNotFoundError:
                continue  # Skip missing tables

    def _export_metadata(self, output_dir: Path) -> None:
        """Export comprehensive metadata as JSON."""
        metadata = {
            'export_info': {
                'format': 'parquet',
                'compression': 'snappy',
                'partitioning': ['country_group', 'decision_year'],
                'source_file': str(self.wide_csv),
                'row_count': len(self.df),
                'column_count': len(self.df.columns)
            },
            'legal_semantics': self.get_legal_metadata(),
            'column_descriptions': self._get_column_descriptions(),
            'data_quality': {
                'schema_echo_cases': int(self.df['schema_echo_flag'].sum()) if 'schema_echo_flag' in self.df.columns else 0,
                'missing_dates': int(self.df['decision_date'].isna().sum()) if 'decision_date' in self.df.columns else 0,
                'zero_fines': int((self.df['fine_eur'] == 0).sum()) if 'fine_eur' in self.df.columns else 0
            },
            'usage_notes': [
                "Use country_group and decision_year for efficient filtering",
                "Boolean columns use pandas nullable boolean dtype",
                "Empty strings converted to null for Parquet compatibility",
                "Consult validation_report.json for data quality issues"
            ]
        }

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)