#!/usr/bin/env python3
"""
GDPR Enforcement Data Schema Validation
=======================================

This module defines and validates the schema for GDPR enforcement data using pandera.
It provides data type validation, value range checks, and custom validation rules.

Author: Enhanced Data Preprocessing Pipeline
Date: 2025-09-20
"""

import pandas as pd
import os
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from datetime import datetime
import re
from typing import Optional
import argparse

# Define allowed values for categorical fields
COUNTRY_CODES = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                'FR', 'GR', 'HU', 'IE', 'IS', 'IT', 'LI', 'LT', 'LU', 'LV', 'MT',
                'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'EU', 'UNKNOWN']

CASE_TRIGGERS = ['COMPLAINT', 'BREACH_NOTIFICATION', 'EX_OFFICIO_INVESTIGATION', 'OTHER', 'UNKNOWN']

YES_NO_UNKNOWN = ['Y', 'N', 'UNKNOWN']

DPA_ROLES = ['Lead_Authority', 'Concerned_Authority', 'N_A']

APPEAL_SUCCESS = ['Successful', 'Unsuccessful', 'N_A']

PRIOR_INFRINGEMENTS = ['PRIOR_INFRINGER_PRESENT', 'DISCUSSED_NO_PRIOR_FOUND', 'NOT_DISCUSSED', 'UNKNOWN']

DEFENDANT_ROLES = ['Controller', 'Processor', 'Joint_Controller', 'UNKNOWN']

DEFENDANT_CATEGORIES = ['PUBLIC_AUTHORITY', 'BUSINESS', 'NON_PROFIT', 'INDIVIDUAL', 'OTHER', 'UNKNOWN']

SENSITIVE_DATA_TYPES = ['Health', 'Biometric', 'Genetic', 'Racial_Ethnic', 'Political',
                       'Religious', 'Philosophical', 'Trade_Union', 'Sexual', 'Criminal', 'Other', 'N_A']

VULNERABLE_SUBJECTS = ['Children', 'Elderly', 'Employees', 'Patients', 'Students',
                      'Other_Vulnerable', 'None_Mentioned', 'UNKNOWN']

LEGAL_BASIS = ['Consent', 'Contract', 'Legal_Obligation', 'Vital_Interests',
               'Public_Task', 'Legitimate_Interest', 'Not_Specified', 'UNKNOWN']

TRANSFER_MECHANISMS = ['Adequacy_Decision', 'Standard_Contractual_Clauses',
                      'Binding_Corporate_Rules', 'Certification', 'Derogation', 'None_Specified', 'N_A']

ASSESSMENT_VALUES = ['Y', 'N', 'Not_Assessed', 'Not_Evaluated', 'Not_Discussed', 'UNKNOWN']

DAMAGE_TYPES = ['Material_Damage', 'Non_Material_Damage', 'Both', 'None', 'Not_Assessed', 'UNKNOWN']

DPIA_STATUS = ['Required_And_Conducted', 'Required_Not_Conducted', 'Not_Required', 'Not_Assessed', 'UNKNOWN']

SECURITY_STATUS = ['Adequate', 'Inadequate', 'Not_Evaluated', 'UNKNOWN']

COMPLIANCE_STATUS = ['Compliant', 'Violated', 'Not_Evaluated', 'UNKNOWN']

RIGHTS_TYPES = ['Access', 'Rectification', 'Erasure', 'Portability', 'Objection',
               'Restrict_Processing', 'Automated_Decision_Making', 'N_A']

COOPERATION_LEVELS = ['Cooperative', 'Partially_Cooperative', 'Uncooperative', 'Not_Discussed', 'UNKNOWN']

MITIGATING_TIMING = ['Before_Investigation', 'During_Investigation', 'None', 'Not_Discussed', 'UNKNOWN']

PRIOR_MEASURES = ['Mitigating_Factor', 'Aggravating_Factor', 'Neutral', 'Not_Considered', 'N_A']

SANCTION_TYPES = ['Fine', 'Warning', 'Reprimand', 'Compliance_Order', 'Processing_Ban_Temporary',
                 'Processing_Ban_Permanent', 'Data_Flow_Suspension', 'Other_Corrective_Measure', 'None']

FINE_CALCULATION = ['Y', 'Partially', 'N', 'N_A']

CURRENCY_CODES = ['EUR', 'NOK', 'SEK', 'DKK', 'PLN', 'CZK', 'HUF', 'N_A']

def validate_date_format(date_str: str) -> bool:
    """Validate DD-MM-YYYY date format."""
    if pd.isna(date_str) or date_str == 'UNKNOWN':
        return True
    try:
        datetime.strptime(str(date_str), '%d-%m-%Y')
        return True
    except ValueError:
        return False

def validate_gdpr_articles(article_str: str) -> bool:
    """Validate GDPR article references format."""
    if pd.isna(article_str) or article_str in ['UNKNOWN', 'None', '']:
        return True

    # Pattern to match "Art. X" or "Art. X(Y)" or "Art. X(Y)(Z)"
    pattern = r'Art\.\s*\d+(?:\(\d+\))?(?:\([a-z]\))?'

    # Split by comma and check each article
    articles = [art.strip() for art in str(article_str).split(',')]
    for article in articles:
        if not re.match(pattern, article):
            return False
    return True

def validate_multi_select(value: str, allowed_values: list) -> bool:
    """Validate comma-separated multi-select fields."""
    if pd.isna(value) or value in ['UNKNOWN', 'N_A', 'None_Mentioned', '']:
        return True

    # Handle list format like "[Children,Employees]"
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]

    # Split by comma and validate each value
    values = [v.strip() for v in str(value).split(',')]
    for v in values:
        if v not in allowed_values:
            return False
    return True

def validate_subjects_affected(value) -> bool:
    """Validate subjects affected field (integer, range, or UNKNOWN)."""
    if pd.isna(value) or value == 'UNKNOWN':
        return True

    # Check if it's a number
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        pass

    # Check if it's a range format like "1000-5000"
    range_pattern = r'^\d+-\d+$'
    if re.match(range_pattern, str(value)):
        return True

    return False

# Define the schema
gdpr_schema = DataFrameSchema({
    "ID": Column(str, nullable=False, unique=True),

    "A1_Country": Column(str, Check.isin(COUNTRY_CODES), nullable=False),

    "A2_Authority": Column(str, nullable=False),

    "A3_DecisionDate": Column(str, Check(validate_date_format,
                             error="Invalid date format. Expected DD-MM-YYYY or UNKNOWN")),

    "A4_CaseTrigger": Column(str, Check.isin(CASE_TRIGGERS)),

    "A5_CrossBorder": Column(str, Check.isin(YES_NO_UNKNOWN)),

    "A6_DPARole": Column(str, Check.isin(DPA_ROLES)),

    "A7_IsAppeal": Column(str, Check.isin(['Y', 'N'])),

    "A8_AppealSuccess": Column(str, Check.isin(APPEAL_SUCCESS)),

    "A9_PriorInfringements": Column(str, Check.isin(PRIOR_INFRINGEMENTS)),

    "A10_DefendantCount": Column(str, Check(lambda x: pd.isna(x) or x == 'UNKNOWN' or
                                           (str(x).isdigit() and int(x) >= 1),
                                 error="Must be positive integer or UNKNOWN")),

    "A11_DefendantName": Column(str),

    "A12_DefendantRole": Column(str, Check.isin(DEFENDANT_ROLES)),

    "A13_InstitutionalIdentity": Column(str),

    "A14_EconomicSector": Column(str),

    "A15_DefendantCategory": Column(str, Check.isin(DEFENDANT_CATEGORIES)),

    "A16_SensitiveData": Column(str, Check.isin(YES_NO_UNKNOWN)),

    "A17_SensitiveDataTypes": Column(str, Check(lambda x: validate_multi_select(x, SENSITIVE_DATA_TYPES),
                                      error="Invalid sensitive data types")),

    "A18_Cookies": Column(str, Check.isin(YES_NO_UNKNOWN)),

    "A19_VulnerableSubjects": Column(str, Check(lambda x: validate_multi_select(x, VULNERABLE_SUBJECTS),
                                      error="Invalid vulnerable subjects")),

    "A20_LegalBasis": Column(str, Check(lambda x: validate_multi_select(x, LEGAL_BASIS),
                             error="Invalid legal basis values")),

    "A21_DataTransfers": Column(str, Check.isin(YES_NO_UNKNOWN)),

    "A22_TransferMechanism": Column(str, Check(lambda x: validate_multi_select(x, TRANSFER_MECHANISMS),
                                     error="Invalid transfer mechanisms")),

    "A23_HighRiskProcessing": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A24_PowerImbalance": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A25_SubjectsAffected": Column(str, Check(validate_subjects_affected,
                                   error="Must be integer, range (1000-5000), or UNKNOWN")),

    "A26_InfringementDuration": Column(str),

    "A27_DamageEstablished": Column(str, Check.isin(DAMAGE_TYPES), nullable=True),

    "A28_DPIARequired": Column(str, Check.isin(DPIA_STATUS)),

    "A29_SecurityMeasures": Column(str, Check.isin(SECURITY_STATUS)),

    "A30_DataMinimization": Column(str, Check.isin(COMPLIANCE_STATUS)),

    "A31_SubjectRightsRequests": Column(str, Check.isin(YES_NO_UNKNOWN)),

    "A32_RightsInvolved": Column(str, Check(lambda x: validate_multi_select(x, RIGHTS_TYPES),
                                 error="Invalid rights types")),

    "A33_TransparencyObligations": Column(str, Check.isin(COMPLIANCE_STATUS)),

    "A34_GDPREvaluated": Column(str, Check(validate_gdpr_articles,
                                error="Invalid GDPR article format")),

    "A35_GDPRViolated": Column(str, Check(validate_gdpr_articles,
                               error="Invalid GDPR article format"), nullable=True),

    "A36_NoInfringement": Column(str, Check.isin(['Y', 'N'])),

    "A37_LegalBasisInvalid": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A38_NegligenceEstablished": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A39_ManagementAuthorization": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A40_DefendantCooperation": Column(str, Check.isin(COOPERATION_LEVELS)),

    "A41_MitigatingMeasures": Column(str, Check.isin(MITIGATING_TIMING), nullable=True),

    "A42_PriorNonCompliance": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A43_FinancialBenefit": Column(str, Check.isin(ASSESSMENT_VALUES)),

    "A44_PriorMeasuresConsidered": Column(str, Check.isin(PRIOR_MEASURES)),

    "A45_SanctionType": Column(str, Check(lambda x: validate_multi_select(x, SANCTION_TYPES),
                               error="Invalid sanction types"), nullable=True),

    "A46_FineAmount": Column(str, Check(lambda x: pd.isna(x) or x in ['N_A', 'UNKNOWN'] or
                                       (str(x).replace('.', '').isdigit() and float(x) >= 0),
                             error="Must be non-negative number, N_A, or UNKNOWN")),

    "A47_FineCurrency": Column(str, Check.isin(CURRENCY_CODES)),

    "A48_EDPBGuidelines": Column(str, Check.isin(YES_NO_UNKNOWN)),

    "A49_FineCalculationFactors": Column(str, Check.isin(FINE_CALCULATION)),

    "A50_CaseSummary": Column(str)
})

def validate_dataframe(df: pd.DataFrame, schema: DataFrameSchema = gdpr_schema) -> tuple:
    """
    Validate a DataFrame against the GDPR schema.

    Returns:
        tuple: (is_valid: bool, validation_errors: list)
    """
    try:
        validated_df = schema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        return False, e.failure_cases.to_dict('records')

def generate_validation_report(df: pd.DataFrame, output_file: str = None) -> dict:
    """
    Generate a comprehensive validation report.

    Args:
        df: DataFrame to validate
        output_file: Optional file path to save the report

    Returns:
        dict: Validation report
    """
    print("Running Schema Validation...")
    print("="*50)

    is_valid, errors = validate_dataframe(df)

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'is_valid': is_valid,
        'error_count': len(errors) if len(errors) > 0 else 0,
        'errors_by_column': {},
        'errors_by_check': {},
        'sample_errors': []
    }

    if is_valid:
        print("✓ Schema validation passed successfully!")
        print(f"  Validated {len(df)} rows across {len(df.columns)} columns")
    else:
        print(f"✗ Schema validation failed with {len(errors)} errors")

        # Group errors by column
        for error in errors:
            column = error.get('column', 'Unknown')
            check = error.get('check', 'Unknown')

            if column not in report['errors_by_column']:
                report['errors_by_column'][column] = 0
            report['errors_by_column'][column] += 1

            if check not in report['errors_by_check']:
                report['errors_by_check'][check] = 0
            report['errors_by_check'][check] += 1

            # Keep sample errors (first 10)
            if len(report['sample_errors']) < 10:
                report['sample_errors'].append({
                    'column': column,
                    'check': check,
                    'row_index': error.get('index', 'Unknown'),
                    'invalid_value': error.get('failure_case', 'Unknown')
                })

        # Display error summary
        print(f"\nErrors by column:")
        for column, count in sorted(report['errors_by_column'].items()):
            print(f"  {column}: {count} errors")

        print(f"\nSample validation errors:")
        for i, error in enumerate(report['sample_errors'][:5], 1):
            print(f"  {i}. Column '{error['column']}' at row {error['row_index']}: "
                  f"'{error['invalid_value']}'")

    # Save report if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📁 Validation report saved to: {output_file}")

    return report

def main():
    """Main validation function."""
    print("GDPR Data Schema Validation")
    print("="*50)

    # CLI args
    parser = argparse.ArgumentParser(description="GDPR Schema Validation")
    parser.add_argument('--input', '-i', dest='input_path', default='dataNorway.csv', help='Path to input CSV file')
    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.input_path)
        print(f"✓ Loaded data from {args.input_path}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data '{args.input_path}': {e}")
        return

    # Generate validation report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'schema_validation_report_{timestamp}.json'

    report = generate_validation_report(df, report_file)

    # Provide recommendations
    if not report['is_valid']:
        print(f"\n📋 Recommendations:")
        print(f"  1. Review and fix {report['error_count']} validation errors")
        print(f"  2. Most problematic columns: {', '.join(list(report['errors_by_column'].keys())[:3])}")
        print(f"  3. Implement data cleaning for categorical value standardization")
        print(f"  4. Consider updating schema if business rules have changed")

if __name__ == "__main__":
    main()