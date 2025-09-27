from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import pyarrow as pa
    import pyarrow.feather as feather
    import pandas as pd
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from .base import BaseExporter


class ArrowExporter(BaseExporter):
    """Export data to Arrow/Feather format for cross-language compatibility."""

    def __init__(self, wide_csv: Path, long_tables_dir: Optional[Path] = None):
        if not PYARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for Arrow export. Install with: pip install pyarrow"
            )
        super().__init__(wide_csv, long_tables_dir)

    def export(self, output_dir: Path, compression: str = 'zstd') -> None:
        """Export data to Arrow/Feather format."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export main wide dataset
        self._export_wide_data(output_dir, compression)

        # Export long tables if available
        if self.long_tables_dir:
            self._export_long_tables(output_dir, compression)

        # Export metadata and schema information
        self._export_metadata(output_dir)
        self._export_arrow_schema(output_dir)

    def _export_wide_data(self, output_dir: Path, compression: str) -> None:
        """Export wide dataset as Feather file."""
        df = self.df.copy()

        # Prepare for Arrow format
        df = self._prepare_for_arrow(df)

        # Create Arrow table with metadata
        table = self._create_arrow_table_with_metadata(df)

        # Write to Feather format
        feather.write_feather(
            table,
            output_dir / "wide_data.feather",
            compression=compression
        )

        # Also write as Arrow IPC format for maximum compatibility
        with pa.OSFile(str(output_dir / "wide_data.arrow"), 'wb') as sink:
            with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
                writer.write_table(table)

    def _prepare_for_arrow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Arrow format with optimal types."""
        df = df.copy()

        # Convert string columns to Arrow string type
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            # Convert empty strings to None
            df[col] = df[col].replace('', None)
            df[col] = df[col].astype('string')

        # Optimize categorical columns
        categorical_optimizations = {
            'country_code': None,  # Keep existing categories
            'country_group': ['EU', 'EEA', 'NON_EEA'],
            'country_status': ['DISCUSSED', 'NOT_DISCUSSED'],
            'decision_date_status': ['DISCUSSED', 'NOT_DISCUSSED', 'PARSE_ERROR'],
            'fine_status': ['DISCUSSED', 'NOT_MENTIONED', 'PARSE_ERROR'],
            'turnover_status': ['DISCUSSED', 'NOT_MENTIONED', 'PARSE_ERROR'],
            'decision_quarter': ['Q1', 'Q2', 'Q3', 'Q4']
        }

        for col, categories in categorical_optimizations.items():
            if col in df.columns:
                if categories:
                    df[col] = pd.Categorical(df[col], categories=categories)
                else:
                    df[col] = df[col].astype('category')

        # Ensure proper numeric types
        numeric_mappings = {
            'fine_eur': 'float64',
            'turnover_eur': 'float64',
            'fine_to_turnover_ratio': 'float64',
            'fine_log1p': 'float64',
            'turnover_log1p': 'float64',
            'decision_year': 'Int64',
            'ingest_line_count': 'Int64',
            'ingest_question_count': 'Int64',
            'n_principles_discussed': 'Int64',
            'n_principles_violated': 'Int64',
            'n_corrective_measures': 'Int64'
        }

        for col, dtype in numeric_mappings.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)

        # Convert boolean columns properly
        bool_cols = [col for col in df.columns if df[col].dtype == 'boolean']
        for col in bool_cols:
            df[col] = df[col].astype('boolean')

        return df

    def _create_arrow_table_with_metadata(self, df: pd.DataFrame) -> pa.Table:
        """Create Arrow table with comprehensive metadata."""
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Prepare metadata
        legal_metadata = self.get_legal_metadata()
        column_descriptions = self._get_column_descriptions()

        # Arrow metadata (all values must be strings)
        arrow_metadata = {
            'title': 'GDPR DPA Decisions Analysis Dataset',
            'description': 'Structured dataset of GDPR enforcement decisions with 68-field questionnaire responses',
            'source': str(self.wide_csv),
            'format': 'arrow_feather',
            'schema_version': '1.0',
            'legal_framework': 'EU GDPR (Regulation 2016/679)',
            'questionnaire_version': '68_questions',
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'row_count': str(len(df)),
            'column_count': str(len(df.columns)),
            'legal_semantics': json.dumps(legal_metadata),
            'column_descriptions': json.dumps(column_descriptions),
            'cross_language_compatibility': json.dumps({
                'python': 'Use pandas.read_feather() or pyarrow.feather.read_table()',
                'r': 'Use arrow::read_feather() or feather::read_feather()',
                'julia': 'Use Arrow.Table() or Feather.read()',
                'rust': 'Use arrow::read::parquet crate'
            }),
            'usage_warnings': json.dumps([
                'Preserve typed missingness semantics when analyzing',
                'NOT_DISCUSSED != NOT_VIOLATED in legal interpretation',
                'Boolean columns use pandas nullable boolean dtype',
                'Consult metadata.json for data quality flags'
            ])
        }

        # Add field-level metadata
        field_metadata = {}
        for i, field in enumerate(table.schema):
            field_name = field.name
            field_meta = {'description': column_descriptions.get(field_name, '')}

            # Add legal interpretation notes for key fields
            if field_name.endswith('_status'):
                field_meta['legal_note'] = 'Preserves typed missingness from questionnaire'
            elif field_name.startswith(('q30_', 'q31_', 'q32_', 'q33_', 'q34_', 'q53_', 'q56_', 'q57_')):
                field_meta['legal_note'] = 'Binary indicator from multi-select question'
            elif field_name in ['fine_eur', 'turnover_eur']:
                field_meta['unit'] = 'EUR'
                field_meta['legal_note'] = 'Zero values may be meaningful (no fine imposed)'

            field_metadata[field_name] = field_meta

        arrow_metadata['field_metadata'] = json.dumps(field_metadata)

        # Create new schema with metadata
        schema_with_metadata = table.schema.with_metadata(arrow_metadata)
        return table.cast(schema_with_metadata)

    def _get_column_descriptions(self) -> Dict[str, str]:
        """Get comprehensive column descriptions for Arrow metadata."""
        base_descriptions = {
            'decision_id': 'Unique identifier for each DPA decision',
            'country_code': 'ISO 3166-1 alpha-2 country code of deciding authority',
            'country_group': 'Geographic classification (EU/EEA/NON_EEA)',
            'dpa_name_canonical': 'Standardized name of data protection authority',
            'decision_date': 'Date of decision issuance (ISO 8601 format)',
            'decision_year': 'Year of decision for temporal analysis',
            'decision_quarter': 'Quarter of year (Q1-Q4) for seasonal analysis',
            'fine_eur': 'Administrative fine amount in EUR (0 = no fine)',
            'turnover_eur': 'Annual turnover mentioned in decision (EUR)',
            'fine_positive': 'Boolean: administrative fine > 0 imposed',
            'fine_log1p': 'Log(1 + fine_eur) for statistical analysis',
            'turnover_log1p': 'Log(1 + turnover_eur) for statistical analysis',
            'fine_to_turnover_ratio': 'Fine as proportion of annual turnover',
            'breach_case': 'Boolean: data breach incident discussed',
            'severity_measures_present': 'Boolean: severe corrective measures imposed',
            'remedy_only_case': 'Boolean: corrective measures but no fine',
            'isic_code': 'ISIC Rev.4 economic activity classification',
            'isic_section': 'High-level economic sector from ISIC',
            'schema_echo_flag': 'Boolean: AI response contains schema artifacts'
        }

        # Add questionnaire text fields
        text_fields = {
            'q36_text': 'Summary of Article 5/6 findings (normalized text)',
            'q52_text': 'Fine calculation reasoning (normalized text)',
            'q67_text': 'EDPB guidelines references (normalized text)',
            'q68_text': 'Expert case summary (normalized text)'
        }
        base_descriptions.update(text_fields)

        return base_descriptions

    def _export_long_tables(self, output_dir: Path, compression: str) -> None:
        """Export long tables as Arrow/Feather files."""
        long_output = output_dir / "long_tables"
        long_output.mkdir(exist_ok=True)

        tables = [
            'defendant_classifications.csv', 'breach_types.csv',
            'special_data_categories.csv', 'mitigating_actions.csv',
            'article_5_discussed.csv', 'article_5_violated.csv',
            'article_6_discussed.csv', 'legal_basis_relied_on.csv',
            'consent_issues.csv', 'li_test_outcome.csv',
            'corrective_powers.csv',
            'rights_discussed.csv', 'rights_violated.csv',
            'aggravating_factors.csv', 'mitigating_factors.csv'
        ]

        for table_name in tables:
            try:
                df = self.load_long_table(table_name)

                # Optimize for Arrow
                df['decision_id'] = df['decision_id'].astype('string')
                df['option'] = df['option'].astype('string')
                df['status'] = df['status'].astype('category')
                df['token_status'] = df['token_status'].astype('category')

                # Create Arrow table with metadata
                table = pa.Table.from_pandas(df, preserve_index=False)

                # Add table-specific metadata
                table_metadata = {
                    'table_type': 'long_format_multi_select',
                    'source_question': table_name.split('.')[0].replace('_', ' ').title(),
                    'description': f'Long-format expansion of {table_name}',
                    'legal_note': 'Each row represents one selected option per decision'
                }

                schema_with_metadata = table.schema.with_metadata(table_metadata)
                table = table.cast(schema_with_metadata)

                # Write Feather file
                output_file = long_output / table_name.replace('.csv', '.feather')
                feather.write_feather(table, output_file, compression=compression)

            except FileNotFoundError:
                continue

    def _export_metadata(self, output_dir: Path) -> None:
        """Export comprehensive metadata as JSON."""
        metadata = {
            'format_info': {
                'primary_format': 'feather',
                'secondary_format': 'arrow_ipc',
                'compression': 'zstd',
                'cross_language': True,
                'zero_copy_reads': True
            },
            'dataset_info': {
                'title': 'GDPR DPA Decisions Analysis Dataset',
                'description': 'Structured enforcement decisions with 68-field questionnaire',
                'legal_framework': 'EU GDPR (Regulation 2016/679)',
                'row_count': len(self.df),
                'column_count': len(self.df.columns),
                'source_file': str(self.wide_csv)
            },
            'legal_semantics': self.get_legal_metadata(),
            'column_descriptions': self._get_column_descriptions(),
            'cross_language_usage': {
                'python': [
                    "import pandas as pd",
                    "df = pd.read_feather('wide_data.feather')",
                    "# Rich metadata preserved in Arrow table"
                ],
                'r': [
                    "library(arrow)",
                    "df <- read_feather('wide_data.feather')",
                    "# Or: tbl <- arrow::read_feather('wide_data.feather')"
                ],
                'julia': [
                    "using Arrow, DataFrames",
                    "df = DataFrame(Arrow.Table('wide_data.feather'))"
                ]
            },
            'performance_notes': [
                "Zero-copy reads in supported languages",
                "Categorical columns optimized for memory efficiency",
                "Compression reduces file size by ~70% vs CSV",
                "Boolean columns use efficient nullable representation"
            ]
        }

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    def _export_arrow_schema(self, output_dir: Path) -> None:
        """Export Arrow schema information for cross-language compatibility."""
        df = self._prepare_for_arrow(self.df.copy())
        table = pa.Table.from_pandas(df, preserve_index=False)

        schema_info = {
            'arrow_schema': str(table.schema),
            'field_types': {
                field.name: str(field.type) for field in table.schema
            },
            'pandas_dtypes': {
                col: str(dtype) for col, dtype in df.dtypes.items()
            },
            'nullable_columns': [
                field.name for field in table.schema
                if field.nullable
            ],
            'categorical_columns': [
                col for col in df.columns
                if pd.api.types.is_categorical_dtype(df[col])
            ]
        }

        with open(output_dir / "arrow_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema_info, f, indent=2, ensure_ascii=False)