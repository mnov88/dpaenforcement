#!/usr/bin/env python3
"""
GDPR Enforcement Data Quality Assessment
========================================

This script performs comprehensive data quality assessment on the GDPR enforcement dataset,
identifying missing values, inconsistencies, outliers, and data quality issues.

Author: Enhanced Data Preprocessing Pipeline
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the CSV data with proper error handling."""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def assess_missing_values(df):
    """Analyze missing values and patterns."""
    print("\n" + "="*60)
    print("MISSING VALUE ANALYSIS")
    print("="*60)

    # Count missing values per column
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Data_Type': df.dtypes
    })

    # Sort by missing percentage
    missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)

    print("\nColumns with missing values:")
    high_missing = missing_stats[missing_stats['Missing_Percentage'] > 0]
    if len(high_missing) > 0:
        print(high_missing.to_string(index=False))
    else:
        print("No missing values found!")

    # Identify rows with high missing values
    row_missing_counts = df.isnull().sum(axis=1)
    high_missing_rows = row_missing_counts[row_missing_counts > 10]

    if len(high_missing_rows) > 0:
        print(f"\nRows with >10 missing values:")
        for idx, count in high_missing_rows.items():
            row_id = df.iloc[idx]['ID'] if 'ID' in df.columns else idx
            print(f"  Row {idx} (ID: {row_id}): {count} missing values")

    return missing_stats

def assess_data_consistency(df):
    """Check for data consistency issues."""
    print("\n" + "="*60)
    print("DATA CONSISTENCY ANALYSIS")
    print("="*60)

    issues = []

    # Check for UNKNOWN vs other missing representations
    unknown_patterns = ['UNKNOWN', 'N_A', '', 'None', 'UNDEFINED']
    for col in df.columns:
        if df[col].dtype == 'object':
            for pattern in unknown_patterns:
                count = (df[col] == pattern).sum()
                if count > 0:
                    issues.append(f"Column {col}: {count} '{pattern}' values")

    # Check for bracket inconsistencies (e.g., [NO] vs NO)
    if 'A1_Country' in df.columns:
        country_values = df['A1_Country'].value_counts()
        print(f"\nCountry value distribution:")
        print(country_values.to_string())

        # Check for bracket formatting
        bracketed = df['A1_Country'].str.contains(r'\[.*\]', na=False).sum()
        if bracketed > 0:
            issues.append(f"A1_Country: {bracketed} values with inconsistent bracket formatting")

    # Check date format consistency
    if 'A3_DecisionDate' in df.columns:
        date_col = df['A3_DecisionDate'].dropna()
        unknown_dates = (date_col == 'UNKNOWN').sum()
        if unknown_dates > 0:
            issues.append(f"A3_DecisionDate: {unknown_dates} UNKNOWN date values")

        # Try to parse dates
        parsed_dates = 0
        for date_str in date_col:
            if date_str != 'UNKNOWN':
                try:
                    pd.to_datetime(date_str, format='%d-%m-%Y')
                    parsed_dates += 1
                except:
                    issues.append(f"A3_DecisionDate: Invalid date format '{date_str}'")

    # Check fine amount consistency
    if 'A46_FineAmount' in df.columns:
        fines = df['A46_FineAmount'].dropna()
        zero_fines = (fines == 0).sum()
        non_numeric = pd.to_numeric(fines, errors='coerce').isna().sum()
        if non_numeric > 0:
            issues.append(f"A46_FineAmount: {non_numeric} non-numeric fine values")

    if issues:
        print("\nConsistency issues found:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ No major consistency issues found")

    return issues

def assess_outliers(df):
    """Identify statistical outliers in numerical fields."""
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS")
    print("="*60)

    numeric_cols = ['A10_DefendantCount', 'A25_SubjectsAffected', 'A46_FineAmount']
    outliers_found = {}

    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, handling non-numeric values
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            clean_data = numeric_data.dropna()

            if len(clean_data) > 0:
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]

                if len(outliers) > 0:
                    outliers_found[col] = outliers
                    print(f"\n{col} outliers (IQR method):")
                    print(f"  Valid range: {lower_bound:.2f} - {upper_bound:.2f}")
                    print(f"  Found {len(outliers)} outliers: {sorted(outliers.values)}")

    return outliers_found

def assess_logical_consistency(df):
    """Check for logical inconsistencies between related fields."""
    print("\n" + "="*60)
    print("LOGICAL CONSISTENCY ANALYSIS")
    print("="*60)

    logical_issues = []

    # Check fine amount vs sanction type consistency
    if 'A46_FineAmount' in df.columns and 'A45_SanctionType' in df.columns:
        fine_amounts = pd.to_numeric(df['A46_FineAmount'], errors='coerce')
        sanction_types = df['A45_SanctionType']

        # Cases where fine amount > 0 but sanction type doesn't include "Fine"
        has_fine = fine_amounts > 0
        mentions_fine = sanction_types.str.contains('Fine', na=False)

        inconsistent_fines = has_fine & ~mentions_fine
        if inconsistent_fines.sum() > 0:
            logical_issues.append(f"Fine amount > 0 but sanction type doesn't mention 'Fine': {inconsistent_fines.sum()} cases")

        # Cases where fine amount = 0 but sanction type includes "Fine"
        no_fine = (fine_amounts == 0) | fine_amounts.isna()
        zero_but_fine = no_fine & mentions_fine
        if zero_but_fine.sum() > 0:
            logical_issues.append(f"Fine amount = 0 but sanction type mentions 'Fine': {zero_but_fine.sum()} cases")

    # Check appeal success vs appeal status
    if 'A7_IsAppeal' in df.columns and 'A8_AppealSuccess' in df.columns:
        is_appeal = df['A7_IsAppeal'] == 'Y'
        appeal_success = df['A8_AppealSuccess'].isin(['Successful', 'Unsuccessful'])

        # Appeals without success status
        appeals_no_outcome = is_appeal & ~appeal_success
        if appeals_no_outcome.sum() > 0:
            logical_issues.append(f"Appeals without success outcome: {appeals_no_outcome.sum()} cases")

        # Non-appeals with success status
        non_appeals_with_outcome = ~is_appeal & appeal_success
        if non_appeals_with_outcome.sum() > 0:
            logical_issues.append(f"Non-appeals with success outcome: {non_appeals_with_outcome.sum()} cases")

    if logical_issues:
        print("\nLogical inconsistencies found:")
        for issue in logical_issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ No logical inconsistencies found")

    return logical_issues

def assess_data_completeness(df):
    """Assess completeness of key data points."""
    print("\n" + "="*60)
    print("DATA COMPLETENESS ASSESSMENT")
    print("="*60)

    # Key fields that should have high completeness
    critical_fields = [
        'A1_Country', 'A2_Authority', 'A3_DecisionDate',
        'A11_DefendantName', 'A15_DefendantCategory', 'A45_SanctionType'
    ]

    completeness_scores = {}
    for field in critical_fields:
        if field in df.columns:
            # Count non-null, non-UNKNOWN values
            valid_values = (~df[field].isna()) & (df[field] != 'UNKNOWN')
            completeness = valid_values.sum() / len(df) * 100
            completeness_scores[field] = completeness

            status = "✓" if completeness >= 90 else "⚠️" if completeness >= 70 else "✗"
            print(f"  {status} {field}: {completeness:.1f}% complete")

    # Overall data quality score per row
    row_scores = []
    for idx, row in df.iterrows():
        non_missing = 0
        total_fields = 0
        for col in df.columns:
            if col != 'ID':  # Exclude ID from quality scoring
                total_fields += 1
                if pd.notna(row[col]) and row[col] not in ['UNKNOWN', 'N_A', '']:
                    non_missing += 1

        score = (non_missing / total_fields * 100) if total_fields > 0 else 0
        row_scores.append(score)

    df_quality = df.copy()
    df_quality['Quality_Score'] = row_scores

    print(f"\nOverall data quality statistics:")
    print(f"  Mean quality score: {np.mean(row_scores):.1f}%")
    print(f"  Median quality score: {np.median(row_scores):.1f}%")
    print(f"  Rows with >80% quality: {(np.array(row_scores) > 80).sum()}")
    print(f"  Rows with <50% quality: {(np.array(row_scores) < 50).sum()}")

    return completeness_scores, row_scores

def generate_summary_report(df, missing_stats, consistency_issues, outliers, logical_issues, completeness_scores):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("DATA QUALITY SUMMARY REPORT")
    print("="*60)

    print(f"Dataset Overview:")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Data size: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print(f"\nData Quality Issues Summary:")
    print(f"  Columns with missing values: {(missing_stats['Missing_Percentage'] > 0).sum()}")
    print(f"  Consistency issues: {len(consistency_issues)}")
    print(f"  Logical inconsistencies: {len(logical_issues)}")
    print(f"  Columns with outliers: {len(outliers)}")

    print(f"\nRecommendations:")

    # Missing value recommendations
    high_missing = missing_stats[missing_stats['Missing_Percentage'] > 50]
    if len(high_missing) > 0:
        print(f"  📋 Consider excluding or carefully handling {len(high_missing)} columns with >50% missing data")

    # Consistency recommendations
    if consistency_issues:
        print(f"  🔧 Standardize formatting for {len(consistency_issues)} consistency issues")

    # Outlier recommendations
    if outliers:
        print(f"  📊 Review and validate {len(outliers)} columns with outliers")

    # Completeness recommendations
    low_completeness = {k: v for k, v in completeness_scores.items() if v < 70}
    if low_completeness:
        print(f"  ⚠️ Improve data collection for {len(low_completeness)} critical fields with <70% completeness")

    print(f"\nNext Steps:")
    print(f"  1. Implement data cleaning pipeline to address identified issues")
    print(f"  2. Establish data validation rules for future data collection")
    print(f"  3. Create standardized missing value handling strategy")
    print(f"  4. Implement data quality monitoring for ongoing data integrity")

def main():
    """Main execution function."""
    print("GDPR Enforcement Data Quality Assessment")
    print("="*60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # CLI args
    parser = argparse.ArgumentParser(description="GDPR Enforcement Data Quality Assessment")
    parser.add_argument('--input', '-i', dest='input_path', default='dataNorway.csv', help='Path to input CSV file')
    args = parser.parse_args()

    # Load data
    df = load_data(args.input_path)
    if df is None:
        return

    # Perform assessments
    missing_stats = assess_missing_values(df)
    consistency_issues = assess_data_consistency(df)
    outliers = assess_outliers(df)
    logical_issues = assess_logical_consistency(df)
    completeness_scores, row_scores = assess_data_completeness(df)

    # Generate summary
    generate_summary_report(df, missing_stats, consistency_issues, outliers, logical_issues, completeness_scores)

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save missing value analysis
    missing_stats.to_csv(f'data_quality_missing_analysis_{timestamp}.csv', index=False)

    # Save quality scores
    quality_df = df[['ID']].copy() if 'ID' in df.columns else pd.DataFrame({'Row': range(len(df))})
    quality_df['Quality_Score'] = row_scores
    quality_df.to_csv(f'data_quality_scores_{timestamp}.csv', index=False)

    print(f"\n📁 Detailed results saved:")
    print(f"  - data_quality_missing_analysis_{timestamp}.csv")
    print(f"  - data_quality_scores_{timestamp}.csv")

    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()