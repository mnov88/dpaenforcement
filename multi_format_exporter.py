#!/usr/bin/env python3
"""
Multi-Format Data Exporter
==========================

This module provides comprehensive export functionality for GDPR enforcement data,
supporting multiple output formats with metadata preservation and version control.

Author: Enhanced Data Preprocessing Pipeline
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class MultiFormatExporter:
    """Comprehensive data exporter supporting multiple formats."""

    def __init__(self, base_path: str = "exports"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.metadata = {}
        self.export_log = []

    def log_export(self, format_type: str, filename: str, rows: int, columns: int, size_mb: float):
        """Log export operations."""
        self.export_log.append({
            'timestamp': datetime.now().isoformat(),
            'format': format_type,
            'filename': filename,
            'rows': rows,
            'columns': columns,
            'size_mb': round(size_mb, 2)
        })

    def generate_metadata(self, df: pd.DataFrame, export_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive metadata for the dataset."""
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_usage_bytes': df.memory_usage(deep=True).sum()
            },
            'data_quality_summary': {
                'missing_values_per_column': df.isnull().sum().to_dict(),
                'total_missing_values': df.isnull().sum().sum(),
                'completeness_percentage': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            },
            'column_statistics': {},
            'export_info': export_info or {},
            'schema_version': '1.0',
            'source_file': (export_info or {}).get('source_file', 'dataNorway.csv'),
            'processing_pipeline': [
                'data_quality_assessment.py',
                'gdpr_schema_validation.py',
                'enhanced_data_cleaning.py',
                'multi_format_exporter.py'
            ]
        }

        # Generate statistics for numerical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            metadata['column_statistics'][col] = {
                'type': 'numeric',
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'unique_values': int(df[col].nunique())
            }

        # Generate statistics for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            metadata['column_statistics'][col] = {
                'type': 'categorical',
                'unique_values': int(df[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'value_distribution': value_counts.head(10).to_dict() if len(value_counts) > 0 else {}
            }

        self.metadata = metadata
        return metadata

    def export_csv(self, df: pd.DataFrame, filename: str, include_index: bool = False) -> str:
        """Export to CSV format."""
        output_path = self.base_path / f"{filename}.csv"
        df.to_csv(output_path, index=include_index, encoding='utf-8')

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        self.log_export('CSV', str(output_path), len(df), len(df.columns), file_size)

        return str(output_path)

    def export_excel(self, df: pd.DataFrame, filename: str, sheet_name: str = 'Data') -> str:
        """Export to Excel format with multiple sheets."""
        output_path = self.base_path / f"{filename}.xlsx"

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Data dictionary sheet
            data_dict = self.create_data_dictionary(df)
            data_dict.to_excel(writer, sheet_name='Data_Dictionary', index=False)

            # Summary statistics sheet
            summary_stats = self.create_summary_statistics(df)
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        self.log_export('Excel', str(output_path), len(df), len(df.columns), file_size)

        return str(output_path)

    def export_parquet(self, df: pd.DataFrame, filename: str, compression: str = 'snappy') -> str:
        """Export to Parquet format for efficient storage and analysis."""
        output_path = self.base_path / f"{filename}.parquet"

        # Convert object columns to string to avoid pyarrow issues
        df_parquet = df.copy()
        for col in df_parquet.select_dtypes(include=['object']).columns:
            df_parquet[col] = df_parquet[col].astype(str)

        # Create PyArrow table with metadata
        table = pa.Table.from_pandas(df_parquet)

        # Add custom metadata (convert numpy types to Python types for JSON serialization)
        metadata_json = json.dumps(self.metadata, default=str)
        table = table.replace_schema_metadata({'gdpr_metadata': metadata_json})

        # Write to parquet
        pq.write_table(table, output_path, compression=compression)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        self.log_export('Parquet', str(output_path), len(df), len(df.columns), file_size)

        return str(output_path)

    def export_json(self, df: pd.DataFrame, filename: str, orient: str = 'records') -> str:
        """Export to JSON format."""
        output_path = self.base_path / f"{filename}.json"

        # Create a comprehensive JSON structure
        json_data = {
            'metadata': self.metadata,
            'data': df.to_dict(orient=orient)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str, ensure_ascii=False)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        self.log_export('JSON', str(output_path), len(df), len(df.columns), file_size)

        return str(output_path)

    def create_data_dictionary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a data dictionary describing each column."""
        # Load field descriptions from README if available
        field_descriptions = self.load_field_descriptions()

        data_dict = []
        for col in df.columns:
            col_info = {
                'Column_Name': col,
                'Data_Type': str(df[col].dtype),
                'Non_Null_Count': int(df[col].notna().sum()),
                'Null_Count': int(df[col].isna().sum()),
                'Unique_Values': int(df[col].nunique()),
                'Description': field_descriptions.get(col, 'No description available')
            }

            # Add sample values for categorical columns
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                unique_vals = df[col].dropna().unique()
                col_info['Sample_Values'] = ', '.join([str(v) for v in unique_vals[:10]])

            data_dict.append(col_info)

        return pd.DataFrame(data_dict)

    def load_field_descriptions(self) -> Dict[str, str]:
        """Load field descriptions from documentation."""
        # This would ideally read from the README or a separate documentation file
        # For now, return basic descriptions based on column patterns
        descriptions = {}

        field_patterns = {
            'A1_Country': 'Country code of the deciding Data Protection Authority',
            'A2_Authority': 'Full name of the deciding authority',
            'A3_DecisionDate': 'Date of the decision (DD-MM-YYYY format)',
            'A4_CaseTrigger': 'What triggered the case (complaint, breach notification, etc.)',
            'A5_CrossBorder': 'Whether this is a cross-border case',
            'A10_DefendantCount': 'Number of defendants in the case',
            'A11_DefendantName': 'Name of the primary defendant',
            'A15_DefendantCategory': 'Category of the defendant (business, public authority, etc.)',
            'A25_SubjectsAffected': 'Number of data subjects affected',
            'A46_FineAmount': 'Fine amount in original currency',
            'A47_FineCurrency': 'Currency of the fine',
            'A50_CaseSummary': 'Brief summary of the case'
        }

        return field_patterns

    def create_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics for the dataset."""
        stats = []

        # Overall dataset statistics
        stats.append({
            'Metric': 'Total Rows',
            'Value': len(df),
            'Description': 'Total number of cases in the dataset'
        })

        stats.append({
            'Metric': 'Total Columns',
            'Value': len(df.columns),
            'Description': 'Total number of data fields'
        })

        stats.append({
            'Metric': 'Completeness Percentage',
            'Value': f"{round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)}%",
            'Description': 'Percentage of non-missing values across all fields'
        })

        # Country distribution
        if 'A1_Country' in df.columns:
            country_counts = df['A1_Country'].value_counts()
            stats.append({
                'Metric': 'Most Common Country',
                'Value': f"{country_counts.index[0]} ({country_counts.iloc[0]} cases)",
                'Description': 'Country with the most enforcement cases'
            })

        # Fine statistics
        if 'A46_FineAmount_EUR' in df.columns:
            fines = pd.to_numeric(df['A46_FineAmount_EUR'], errors='coerce').dropna()
            if len(fines) > 0:
                stats.append({
                    'Metric': 'Average Fine (EUR)',
                    'Value': f"€{fines.mean():,.2f}",
                    'Description': 'Average fine amount in EUR'
                })

                stats.append({
                    'Metric': 'Largest Fine (EUR)',
                    'Value': f"€{fines.max():,.2f}",
                    'Description': 'Largest fine amount in EUR'
                })

        return pd.DataFrame(stats)

    def export_all_formats(self, df: pd.DataFrame, base_filename: str, export_info: Dict[str, Any] = None) -> Dict[str, str]:
        """Export dataset to all supported formats."""
        print("📦 Exporting to multiple formats...")

        # Generate metadata
        self.generate_metadata(df, export_info)

        # Save metadata separately
        metadata_path = self.base_path / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # Export to all formats
        exported_files = {}

        # CSV - Standard format
        print("  📄 Exporting to CSV...")
        exported_files['csv'] = self.export_csv(df, base_filename)

        # Excel - With multiple sheets
        print("  📊 Exporting to Excel...")
        exported_files['excel'] = self.export_excel(df, base_filename)

        # Parquet - Efficient binary format
        print("  🗜️ Exporting to Parquet...")
        exported_files['parquet'] = self.export_parquet(df, base_filename)

        # JSON - With metadata
        print("  🌐 Exporting to JSON...")
        exported_files['json'] = self.export_json(df, base_filename)

        # Save export log
        export_log_path = self.base_path / f"{base_filename}_export_log.json"
        with open(export_log_path, 'w') as f:
            json.dump(self.export_log, f, indent=2, default=str)

        exported_files['metadata'] = str(metadata_path)
        exported_files['export_log'] = str(export_log_path)

        return exported_files

    def create_versioned_export(self, df: pd.DataFrame, version: str, description: str = "", source_file: Optional[str] = None) -> Dict[str, str]:
        """Create a versioned export with semantic versioning."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"gdpr_enforcement_data_v{version}_{timestamp}"

        export_info = {
            'version': version,
            'description': description,
            'export_timestamp': datetime.now().isoformat(),
            'git_hash': None,  # Could be populated from git if available
            'processing_version': '1.0.0',
            'source_file': source_file or 'dataNorway.csv'
        }

        return self.export_all_formats(df, base_filename, export_info)

    def generate_export_summary(self) -> str:
        """Generate a summary of all export operations."""
        if not self.export_log:
            return "No exports performed yet."

        summary = f"""
Export Summary Report
====================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Exports: {len(self.export_log)}

Export Details:
"""

        for export in self.export_log:
            summary += f"""
  {export['format']} Export:
    File: {export['filename']}
    Size: {export['size_mb']} MB
    Dimensions: {export['rows']} rows × {export['columns']} columns
    Timestamp: {export['timestamp']}
"""

        # Calculate total size
        total_size = sum(export['size_mb'] for export in self.export_log)
        summary += f"\nTotal Export Size: {total_size:.2f} MB"

        return summary

def main():
    """Main execution function."""
    print("Multi-Format Data Exporter")
    print("=" * 50)

    # Load cleaned data
    import glob
    cleaned_files = glob.glob('dataNorway_cleaned_*.csv')

    if not cleaned_files:
        print("❌ No cleaned data files found. Please run enhanced_data_cleaning.py first.")
        return

    # Use the most recent cleaned file
    latest_file = max(cleaned_files, key=os.path.getctime)
    print(f"📂 Loading cleaned data from: {latest_file}")

    try:
        df = pd.read_csv(latest_file)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Initialize exporter
    exporter = MultiFormatExporter()

    # Create versioned export
    version = "1.0"
    description = "Initial cleaned and processed GDPR enforcement dataset"

    print(f"\n🚀 Creating versioned export v{version}")
    exported_files = exporter.create_versioned_export(df, version, description, source_file=latest_file)

    # Display results
    print(f"\n✅ Export completed successfully!")
    print(f"\nExported files:")
    for format_type, filepath in exported_files.items():
        file_size = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
        print(f"  📁 {format_type.upper()}: {filepath} ({file_size:.2f} MB)")

    # Generate and save summary
    summary = exporter.generate_export_summary()
    summary_path = exporter.base_path / "export_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"\n📋 Export summary saved to: {summary_path}")

    print(f"\n🎯 All formats ready for:")
    print(f"  📊 Statistical analysis (CSV, Parquet)")
    print(f"  📈 Business reporting (Excel)")
    print(f"  🌐 Web applications (JSON)")
    print(f"  🔄 Data pipeline integration (Parquet)")
    print(f"  📚 Documentation and metadata preservation")

if __name__ == "__main__":
    main()