"""Cross-format validation utilities to ensure data integrity."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


def validate_export_integrity(source_csv: Path, export_dir: Path) -> Dict[str, Any]:
    """Validate that exports preserve data integrity from source CSV.

    Returns validation report with any integrity issues found.
    """
    validation_results = {
        'source_records': 0,
        'source_columns': 0,
        'issues': [],
        'format_checks': {}
    }

    try:
        # Load source data
        source_df = pd.read_csv(source_csv)
        validation_results['source_records'] = len(source_df)
        validation_results['source_columns'] = len(source_df.columns)

        # Check key columns for null counts
        key_columns = ['decision_id', 'country_group', 'decision_year', 'fine_eur', 'breach_case']
        source_nulls = {}

        for col in key_columns:
            if col in source_df.columns:
                source_nulls[col] = source_df[col].isnull().sum()

        # Validate Parquet export if exists
        parquet_dir = export_dir / 'parquet_data'
        if parquet_dir.exists():
            validation_results['format_checks']['parquet'] = _validate_parquet(
                source_df, parquet_dir, source_nulls
            )

        # Validate Arrow export if exists
        arrow_dir = export_dir / 'arrow_data'
        if arrow_dir.exists():
            validation_results['format_checks']['arrow'] = _validate_arrow(
                source_df, arrow_dir, source_nulls
            )

        # Validate ML export if exists
        ml_dir = export_dir / 'ml_ready'
        if ml_dir.exists():
            validation_results['format_checks']['ml'] = _validate_ml(
                source_df, ml_dir, source_nulls
            )

    except Exception as e:
        validation_results['issues'].append(f"Validation error: {e}")

    return validation_results


def _validate_parquet(source_df: pd.DataFrame, parquet_dir: Path, source_nulls: Dict[str, int]) -> Dict[str, Any]:
    """Validate Parquet export maintains data integrity."""
    result = {'status': 'pass', 'issues': []}

    try:
        # Check main parquet file
        main_file = parquet_dir / 'wide_data.parquet'
        if main_file.exists():
            parquet_df = pd.read_parquet(main_file)

            # Check record count
            if len(parquet_df) != len(source_df):
                result['issues'].append(
                    f"Record count mismatch: source={len(source_df)}, parquet={len(parquet_df)}"
                )

            # Check for corruption sentinel values
            for col in ['country_group', 'decision_year']:
                if col in parquet_df.columns:
                    if 'UNKNOWN' in parquet_df[col].astype(str).values:
                        result['issues'].append(f"Found corruption sentinel 'UNKNOWN' in {col}")
                    if '-999' in parquet_df[col].astype(str).values:
                        result['issues'].append(f"Found corruption sentinel '-999' in {col}")

        if result['issues']:
            result['status'] = 'fail'

    except Exception as e:
        result['status'] = 'error'
        result['issues'].append(f"Parquet validation error: {e}")

    return result


def _validate_arrow(source_df: pd.DataFrame, arrow_dir: Path, source_nulls: Dict[str, int]) -> Dict[str, Any]:
    """Validate Arrow export maintains data integrity."""
    result = {'status': 'pass', 'issues': []}

    try:
        import pyarrow.feather as feather

        arrow_file = arrow_dir / 'wide_data.feather'
        if arrow_file.exists():
            arrow_df = feather.read_feather(arrow_file)

            # Check record count
            if len(arrow_df) != len(source_df):
                result['issues'].append(
                    f"Record count mismatch: source={len(source_df)}, arrow={len(arrow_df)}"
                )

            # Check null preservation for key columns
            for col, source_null_count in source_nulls.items():
                if col in arrow_df.columns:
                    arrow_null_count = arrow_df[col].isnull().sum()
                    if arrow_null_count != source_null_count:
                        result['issues'].append(
                            f"Null count mismatch in {col}: source={source_null_count}, arrow={arrow_null_count}"
                        )

        if result['issues']:
            result['status'] = 'fail'

    except ImportError:
        result['status'] = 'skip'
        result['issues'].append("PyArrow not available")
    except Exception as e:
        result['status'] = 'error'
        result['issues'].append(f"Arrow validation error: {e}")

    return result


def _validate_ml(source_df: pd.DataFrame, ml_dir: Path, source_nulls: Dict[str, int]) -> Dict[str, Any]:
    """Validate ML export preserves targets and legal semantics."""
    result = {'status': 'pass', 'issues': []}

    try:
        full_dataset = ml_dir / 'full_dataset.csv'
        if full_dataset.exists():
            ml_df = pd.read_csv(full_dataset)

            # Check for target preservation (should have missing indicators, not imputed)
            target_cols = [col for col in ml_df.columns if 'fine_eur' in col or 'turnover_eur' in col]

            for col in target_cols:
                if col in ml_df.columns and not col.endswith('_missing'):
                    # Target columns should still have nulls, not be median-imputed
                    if ml_df[col].isnull().sum() == 0 and source_df.get(col, pd.Series()).isnull().sum() > 0:
                        result['issues'].append(f"Target {col} appears to be imputed (no nulls found)")

            # Check for missing indicators
            missing_indicators = [col for col in ml_df.columns if col.endswith('_missing')]
            if not missing_indicators:
                result['issues'].append("No missing indicators found - targets may be imputed")

        if result['issues']:
            result['status'] = 'fail'

    except Exception as e:
        result['status'] = 'error'
        result['issues'].append(f"ML validation error: {e}")

    return result


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """Print human-readable validation report."""
    print("🔍 EXPORT VALIDATION REPORT")
    print("=" * 30)
    print(f"Source: {validation_results['source_records']:,} records, {validation_results['source_columns']:,} columns")

    if validation_results['issues']:
        print(f"\n❌ Global Issues ({len(validation_results['issues'])}):")
        for issue in validation_results['issues']:
            print(f"  • {issue}")

    print("\n📋 Format-Specific Checks:")
    for format_name, result in validation_results['format_checks'].items():
        status_icon = {'pass': '✅', 'fail': '❌', 'error': '🚨', 'skip': '⏭️'}[result['status']]
        print(f"  {status_icon} {format_name.upper()}: {result['status']}")

        if result['issues']:
            for issue in result['issues']:
                print(f"    • {issue}")

    # Overall assessment
    all_passed = all(
        result['status'] in ['pass', 'skip']
        for result in validation_results['format_checks'].values()
    ) and not validation_results['issues']

    if all_passed:
        print("\n🎉 All validations passed! Data integrity preserved across formats.")
    else:
        print("\n⚠️ Some validation issues found. Review export processes.")