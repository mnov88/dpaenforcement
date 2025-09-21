#!/usr/bin/env python3
"""
Enhanced GDPR Data Cleaning Pipeline
====================================

This module provides comprehensive data cleaning functionality for GDPR enforcement data,
addressing inconsistencies, standardizing formats, and preparing data for analysis.

Author: Enhanced Data Preprocessing Pipeline
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import requests
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Optional schema-driven vocabularies for consistent one-hot expansion across files
try:
    from gdpr_schema_validation import (
        SENSITIVE_DATA_TYPES,
        VULNERABLE_SUBJECTS,
        LEGAL_BASIS,
        TRANSFER_MECHANISMS,
        RIGHTS_TYPES,
        SANCTION_TYPES,
    )
except Exception:
    # Fallback minimal sets if schema module not available at runtime
    SENSITIVE_DATA_TYPES = ['Health', 'Biometric', 'Genetic', 'Racial_Ethnic', 'Political',
                            'Religious', 'Philosophical', 'Trade_Union', 'Sexual', 'Criminal', 'Other', 'N_A']
    VULNERABLE_SUBJECTS = ['Children', 'Elderly', 'Employees', 'Patients', 'Students',
                           'Other_Vulnerable', 'None_Mentioned', 'UNKNOWN']
    LEGAL_BASIS = ['Consent', 'Contract', 'Legal_Obligation', 'Vital_Interests',
                   'Public_Task', 'Legitimate_Interest', 'Not_Specified', 'UNKNOWN']
    TRANSFER_MECHANISMS = ['Adequacy_Decision', 'Standard_Contractual_Clauses', 'Binding_Corporate_Rules',
                           'Certification', 'Derogation', 'None_Specified', 'N_A']
    RIGHTS_TYPES = ['Access', 'Rectification', 'Erasure', 'Portability', 'Objection',
                    'Restrict_Processing', 'Automated_Decision_Making', 'N_A']
    SANCTION_TYPES = ['Fine', 'Warning', 'Reprimand', 'Compliance_Order', 'Processing_Ban_Temporary',
                      'Processing_Ban_Permanent', 'Data_Flow_Suspension', 'Other_Corrective_Measure', 'None']

# Fixed top-level GDPR articles to standardize A35_Art_* columns across files
FIXED_A35_TOP_LEVEL = [5, 6, 12, 13, 14, 15, 17, 21, 24, 25, 28, 30, 32, 33, 34, 35]

class GDPRDataCleaner:
    """Enhanced data cleaner for GDPR enforcement dataset."""

    def __init__(self):
        self.cleaning_log = []
        self.currency_rates = {}
        self.statistics = {}

    def log_action(self, action: str, details: str, count: int = 0):
        """Log cleaning actions for audit trail."""
        self.cleaning_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'count': count
        })

    def remove_brackets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove inconsistent bracket formatting from all string columns."""
        print("🔧 Removing bracket inconsistencies...")

        df_clean = df.copy()
        bracket_fixes = 0

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Remove brackets from values like [NO], [Y], [N]
                original_values = df_clean[col].copy()
                df_clean[col] = df_clean[col].astype(str).str.replace(r'^\[(.*)\]$', r'\1', regex=True)

                # Count changes
                changes = (original_values != df_clean[col]).sum()
                if changes > 0:
                    bracket_fixes += changes
                    print(f"  Fixed {changes} bracket values in column {col}")

        self.log_action("remove_brackets", f"Removed brackets from {bracket_fixes} values", bracket_fixes)
        return df_clean

    def standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize various representations of missing values."""
        print("🔧 Standardizing missing values...")

        df_clean = df.copy()
        missing_patterns = ['UNKNOWN', 'N_A', '', 'None', 'UNDEFINED', 'nan', 'NULL']
        total_standardized = 0

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                original_missing = df_clean[col].isna().sum()

                # Replace various missing patterns with NaN
                for pattern in missing_patterns:
                    mask = (df_clean[col] == pattern)
                    df_clean.loc[mask, col] = np.nan

                new_missing = df_clean[col].isna().sum()
                standardized = new_missing - original_missing
                if standardized > 0:
                    total_standardized += standardized
                    print(f"  Standardized {standardized} missing values in column {col}")

        self.log_action("standardize_missing", f"Standardized {total_standardized} missing values", total_standardized)
        return df_clean

    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate date fields."""
        print("🔧 Cleaning date fields...")

        df_clean = df.copy()
        date_columns = ['A3_DecisionDate']

        for col in date_columns:
            if col in df_clean.columns:
                original_dates = df_clean[col].copy()
                valid_dates = 0
                invalid_dates = 0

                # Process each date
                for idx, date_val in df_clean[col].items():
                    if pd.notna(date_val) and str(date_val).strip() != '':
                        date_str = str(date_val).strip()

                        # Skip if already marked as UNKNOWN
                        if date_str == 'UNKNOWN':
                            continue

                        try:
                            # Try to parse DD-MM-YYYY format
                            parsed_date = pd.to_datetime(date_str, format='%d-%m-%Y')
                            df_clean.loc[idx, col] = parsed_date.strftime('%d-%m-%Y')
                            valid_dates += 1
                        except ValueError:
                            # Try other common formats
                            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']:
                                try:
                                    parsed_date = pd.to_datetime(date_str, format=fmt)
                                    df_clean.loc[idx, col] = parsed_date.strftime('%d-%m-%Y')
                                    valid_dates += 1
                                    break
                                except ValueError:
                                    continue
                            else:
                                # Mark as UNKNOWN if unparseable
                                df_clean.loc[idx, col] = 'UNKNOWN'
                                invalid_dates += 1

                print(f"  {col}: {valid_dates} valid dates, {invalid_dates} marked as UNKNOWN")
                self.log_action("clean_dates", f"{col}: {valid_dates} valid, {invalid_dates} invalid", valid_dates + invalid_dates)

        return df_clean

    def fetch_currency_rates(self) -> Dict[str, float]:
        """Fetch historical currency exchange rates (simplified for demo)."""
        # In a real implementation, you'd use a proper API with historical rates
        # For now, using approximate rates
        rates = {
            'NOK': 0.092,  # NOK to EUR
            'SEK': 0.086,  # SEK to EUR
            'DKK': 0.134,  # DKK to EUR
            'PLN': 0.23,   # PLN to EUR
            'CZK': 0.041,  # CZK to EUR
            'HUF': 0.0025, # HUF to EUR
            'EUR': 1.0     # EUR to EUR
        }
        return rates

    def normalize_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize fine amounts to EUR for comparison."""
        print("🔧 Normalizing currency amounts...")

        df_clean = df.copy()

        if 'A46_FineAmount' in df_clean.columns and 'A47_FineCurrency' in df_clean.columns:
            # Add new column for EUR amounts
            df_clean['A46_FineAmount_EUR'] = np.nan

            self.currency_rates = self.fetch_currency_rates()
            conversions = 0

            for idx, row in df_clean.iterrows():
                fine_amount = row['A46_FineAmount']
                currency = row['A47_FineCurrency']

                if pd.notna(fine_amount) and pd.notna(currency) and currency != 'N_A':
                    try:
                        amount = float(fine_amount)
                        if currency in self.currency_rates:
                            eur_amount = amount * self.currency_rates[currency]
                            df_clean.loc[idx, 'A46_FineAmount_EUR'] = round(eur_amount, 2)
                            conversions += 1
                    except (ValueError, TypeError):
                        continue

            print(f"  Converted {conversions} fine amounts to EUR")
            self.log_action("normalize_currency", f"Converted {conversions} amounts to EUR", conversions)

        return df_clean

    def clean_gdpr_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize GDPR article references."""
        print("🔧 Cleaning GDPR article references...")

        df_clean = df.copy()
        article_columns = ['A34_GDPREvaluated', 'A35_GDPRViolated']

        for col in article_columns:
            if col in df_clean.columns:
                cleaned_articles = 0

                for idx, articles in df_clean[col].items():
                    if pd.notna(articles) and articles not in ['UNKNOWN', 'None', '']:
                        article_str = str(articles).strip()

                        # Clean up common formatting issues
                        article_str = re.sub(r'\s+', ' ', article_str)  # Normalize whitespace
                        article_str = re.sub(r',\s*', ', ', article_str)  # Standardize comma spacing

                        # Ensure proper "Art." prefix
                        article_str = re.sub(r'\bArticle\s+(\d+)', r'Art. \1', article_str)
                        article_str = re.sub(r'\bArt\s+(\d+)', r'Art. \1', article_str)

                        if article_str != str(articles):
                            df_clean.loc[idx, col] = article_str
                            cleaned_articles += 1

                print(f"  {col}: Cleaned {cleaned_articles} article references")
                self.log_action("clean_articles", f"{col}: {cleaned_articles} cleaned", cleaned_articles)

        return df_clean

    def expand_a34_top_level_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create boolean indicators for top-level articles in A34_GDPREvaluated (fixed set).

        Encoding: 1 where explicitly present; NaN otherwise.
        """
        print("🔧 Expanding A34_GDPREvaluated into top-level article indicators...")

        df_clean = df.copy()
        source_col = 'A34_GDPREvaluated'
        if source_col not in df_clean.columns:
            return df_clean

        top_level_articles = [str(a) for a in FIXED_A35_TOP_LEVEL]
        article_pattern = re.compile(r"Art\.\s*(\d+)")

        created_cols = 0
        for article in sorted(top_level_articles, key=lambda x: int(x)):
            col_name = f"A34_Art_{article}"
            if col_name not in df_clean.columns:
                df_clean[col_name] = np.nan
                created_cols += 1

        for idx, value in df_clean[source_col].items():
            if pd.notna(value) and str(value).strip() != '':
                found = set(article_pattern.findall(str(value)))
                for a in found:
                    col_name = f"A34_Art_{a}"
                    if col_name in df_clean.columns:
                        df_clean.loc[idx, col_name] = 1

        print(f"  Created {created_cols} A34 article indicator columns")
        self.log_action("expand_a34_articles", f"Created {created_cols} top-level article indicators", created_cols)

        return df_clean

    def normalize_sanction_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize A45_SanctionType tokens: unify separators, trim, deduplicate, sort stable."""
        print("🔧 Normalizing sanction type tokens...")
        df_clean = df.copy()
        col = 'A45_SanctionType'
        if col not in df_clean.columns:
            return df_clean

        normalized = 0
        for idx, val in df_clean[col].items():
            if pd.isna(val) or str(val).strip() in ('', 'UNKNOWN', 'N_A'):
                continue
            s = str(val)
            # unify separators to comma
            s = s.replace('|', ',')
            # split, trim, filter empties
            tokens = [t.strip() for t in s.split(',') if t.strip()]
            # dedupe preserving order
            seen = set()
            uniq = []
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            new_val = ', '.join(uniq)
            if new_val != val:
                df_clean.loc[idx, col] = new_val
                normalized += 1

        print(f"  Normalized {normalized} sanction entries")
        self.log_action("normalize_sanction", f"Normalized {normalized} sanction entries", normalized)

        return df_clean

    def parse_structured_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse A13, A14, numericize A25, and parse A26 duration to months."""
        print("🔧 Parsing structured fields (A13, A14, A25, A26)...")
        df_clean = df.copy()

        # A13: SNA code and description
        if 'A13_InstitutionalIdentity' in df_clean.columns:
            df_clean['A13_SNA_Code'] = np.nan
            df_clean['A13_SNA_Desc'] = np.nan
            for idx, val in df_clean['A13_InstitutionalIdentity'].items():
                if pd.isna(val) or str(val).strip() in ('', 'UNKNOWN'):
                    continue
                s = str(val).strip()
                if ' - ' in s:
                    code, desc = s.split(' - ', 1)
                    df_clean.loc[idx, 'A13_SNA_Code'] = code.strip()
                    df_clean.loc[idx, 'A13_SNA_Desc'] = desc.strip()
                else:
                    # Try pattern S.xx at start
                    m = re.match(r'(S\.?\d+)', s)
                    if m:
                        df_clean.loc[idx, 'A13_SNA_Code'] = m.group(1)
                        df_clean.loc[idx, 'A13_SNA_Desc'] = s[m.end():].lstrip(' -:').strip() or np.nan
                    else:
                        df_clean.loc[idx, 'A13_SNA_Desc'] = s

        # A14: ISIC code, description, level
        if 'A14_EconomicSector' in df_clean.columns:
            df_clean['A14_ISIC_Code'] = np.nan
            df_clean['A14_ISIC_Desc'] = np.nan
            df_clean['A14_ISIC_Level'] = np.nan
            for idx, val in df_clean['A14_EconomicSector'].items():
                if pd.isna(val) or str(val).strip() in ('', 'UNKNOWN'):
                    continue
                s = str(val)
                # Try explicit keys
                mcode = re.search(r'Code\s*:\s*(\d+)', s)
                mdesc = re.search(r'Description\s*:\s*([^,]+)', s)
                mlev = re.search(r'Level\s*:\s*(\d+)', s)
                code = mcode.group(1) if mcode else None
                desc = mdesc.group(1).strip() if mdesc else None
                lev = mlev.group(1) if mlev else None
                if not code:
                    mcode2 = re.search(r'\b(\d{2,5})\b', s)
                    code = mcode2.group(1) if mcode2 else None
                if not desc and code:
                    # take text after code
                    pos = s.find(code)
                    tail = s[pos+len(code):] if pos>=0 else s
                    # strip punctuation
                    desc = tail.strip(' -,:') or None
                if code:
                    df_clean.loc[idx, 'A14_ISIC_Code'] = code
                if desc:
                    df_clean.loc[idx, 'A14_ISIC_Desc'] = desc
                if lev:
                    df_clean.loc[idx, 'A14_ISIC_Level'] = int(lev)

        # A25: numericize
        if 'A25_SubjectsAffected' in df_clean.columns:
            df_clean['A25_SubjectsAffected_min'] = np.nan
            df_clean['A25_SubjectsAffected_max'] = np.nan
            df_clean['A25_SubjectsAffected_midpoint'] = np.nan
            df_clean['A25_SubjectsAffected_is_range'] = 0
            for idx, val in df_clean['A25_SubjectsAffected'].items():
                if pd.isna(val) or str(val).strip() == '' or str(val).strip() == 'UNKNOWN':
                    continue
                s = str(val).replace(',', '')
                if re.match(r'^\d+$', s):
                    v = float(s)
                    df_clean.loc[idx, 'A25_SubjectsAffected_min'] = v
                    df_clean.loc[idx, 'A25_SubjectsAffected_max'] = v
                    df_clean.loc[idx, 'A25_SubjectsAffected_midpoint'] = v
                else:
                    m = re.match(r'^(\d+)-(\d+)$', s)
                    if m:
                        lo = float(m.group(1)); hi = float(m.group(2))
                        df_clean.loc[idx, 'A25_SubjectsAffected_min'] = lo
                        df_clean.loc[idx, 'A25_SubjectsAffected_max'] = hi
                        df_clean.loc[idx, 'A25_SubjectsAffected_midpoint'] = (lo+hi)/2
                        df_clean.loc[idx, 'A25_SubjectsAffected_is_range'] = 1

        # A26: duration months
        if 'A26_InfringementDuration' in df_clean.columns:
            df_clean['A26_Duration_Months'] = np.nan
            for idx, val in df_clean['A26_InfringementDuration'].items():
                if pd.isna(val) or str(val).strip() in ('', 'UNKNOWN'):
                    continue
                s = str(val).lower()
                months = 0.0
                y = re.search(r'(\d+)\s*year', s)
                m = re.search(r'(\d+)\s*month', s)
                w = re.search(r'(\d+)\s*week', s)
                if y:
                    months += int(y.group(1)) * 12
                if m:
                    months += int(m.group(1))
                if w:
                    months += int(w.group(1)) / 4.345
                if months > 0:
                    df_clean.loc[idx, 'A26_Duration_Months'] = round(months, 2)

        return df_clean
    def expand_a35_top_level_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create boolean indicators for top-level articles in A35_GDPRViolated.

        Encoding strategy to avoid skew:
        - 1 where the article is explicitly present in the row
        - NaN otherwise (absence is treated as unknown, not false)
        """
        print("🔧 Expanding A35_GDPRViolated into top-level article indicators...")

        df_clean = df.copy()
        source_col = 'A35_GDPRViolated'

        if source_col not in df_clean.columns:
            return df_clean

        # Use fixed vocabulary for top-level articles to ensure cross-file consistency
        top_level_articles = [str(a) for a in FIXED_A35_TOP_LEVEL]
        article_pattern = re.compile(r"Art\.\s*(\d+)")

        created_cols = 0
        # Initialize columns as NaN (unknown) to avoid implying false
        for article in sorted(top_level_articles, key=lambda x: int(x)):
            col_name = f"A35_Art_{article}"
            if col_name not in df_clean.columns:
                df_clean[col_name] = np.nan
                created_cols += 1

        # Mark presence with 1
        for idx, value in df_clean[source_col].items():
            if pd.notna(value) and str(value).strip() != '':
                found_articles = set(article_pattern.findall(str(value)))
                for article in found_articles:
                    col_name = f"A35_Art_{article}"
                    if col_name in df_clean.columns:
                        df_clean.loc[idx, col_name] = 1

        print(f"  Created {created_cols} A35 article indicator columns")
        self.log_action("expand_a35_articles", f"Created {created_cols} top-level article indicators", created_cols)

        return df_clean

    def convert_yes_no_fields_to_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 0/1 binary variants for Y/N style fields with suffix _bin.

        Mapping:
        - 'Y' -> 1
        - 'N' -> 0
        - others (UNKNOWN, N_A, Partially, empty, NaN) -> NaN
        """
        print("🔧 Converting Y/N fields to binary (_bin) columns...")

        df_clean = df.copy()

        # Columns that are strictly Y/N
        yn_columns = [
            'A7_IsAppeal',
            'A36_NoInfringement'
        ]

        # Columns with Y/N/UNKNOWN (UNKNOWN already standardized to NaN earlier)
        ynu_columns = [
            'A5_CrossBorder',
            'A16_SensitiveData',
            'A18_Cookies',
            'A21_DataTransfers',
            'A31_SubjectRightsRequests',
            'A48_EDPBGuidelines'
        ]

        # Columns with Y/Partially/N/N_A (map Y->1, N->0, others->NaN)
        partial_columns = [
            'A49_FineCalculationFactors'
        ]

        created_cols = 0

        def to_bin(val: Optional[str]) -> float:
            if pd.isna(val):
                return np.nan
            if val == 'Y':
                return 1
            if val == 'N':
                return 0
            return np.nan

        for col_group in [yn_columns, ynu_columns]:
            for col in col_group:
                if col in df_clean.columns:
                    bin_col = f"{col}_bin"
                    if bin_col not in df_clean.columns:
                        df_clean[bin_col] = df_clean[col].apply(to_bin)
                        created_cols += 1

        for col in partial_columns:
            if col in df_clean.columns:
                bin_col = f"{col}_bin"
                if bin_col not in df_clean.columns:
                    df_clean[bin_col] = df_clean[col].apply(to_bin)
                    created_cols += 1

        print(f"  Created {created_cols} binary (_bin) columns")
        self.log_action("convert_yes_no_binary", f"Created {created_cols} binary columns", created_cols)

        return df_clean

    def convert_tri_state_fields_to_pos_bin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create {field}_pos_bin indicating positive vs negative, others -> NaN.

        Mappings:
        - Generic Y/N sets: positive={'Y'}, negative={'N'}
        - A27_DamageEstablished: positive={'Material_Damage','Non_Material_Damage','Both'}, negative={'None'}
        - A28_DPIARequired: positive values startwith 'Required', negative={'Not_Required'}
        - A29_SecurityMeasures: positive={'Adequate'}, negative={'Inadequate'}
        - A30_DataMinimization, A33_TransparencyObligations: positive={'Compliant'}, negative={'Violated'}
        - A40_DefendantCooperation: positive={'Cooperative'}, negative={'Uncooperative'} (Partially -> NaN)
        - A41_MitigatingMeasures: positive={'Before_Investigation','During_Investigation'}, negative={'None'}
        """
        print("🔧 Creating tri-state positive bins (_pos_bin)...")

        df_clean = df.copy()

        def map_generic(val: Optional[str]) -> float:
            if pd.isna(val):
                return np.nan
            if val == 'Y':
                return 1
            if val == 'N':
                return 0
            return np.nan

        field_mappers: Dict[str, callable] = {}
        # Generic Y/N with other statuses
        for col in ['A23_HighRiskProcessing', 'A24_PowerImbalance', 'A37_LegalBasisInvalid',
                    'A38_NegligenceEstablished', 'A39_ManagementAuthorization', 'A42_PriorNonCompliance',
                    'A43_FinancialBenefit']:
            field_mappers[col] = map_generic

        # Specific mappings
        def map_a27(val):
            if pd.isna(val):
                return np.nan
            if val in ['Material_Damage', 'Non_Material_Damage', 'Both']:
                return 1
            if val == 'None':
                return 0
            return np.nan

        def map_a28(val):
            if pd.isna(val):
                return np.nan
            if str(val).startswith('Required'):
                return 1
            if val == 'Not_Required':
                return 0
            return np.nan

        def map_a29(val):
            if pd.isna(val):
                return np.nan
            if val == 'Adequate':
                return 1
            if val == 'Inadequate':
                return 0
            return np.nan

        def map_compliance(val):
            if pd.isna(val):
                return np.nan
            if val == 'Compliant':
                return 1
            if val == 'Violated':
                return 0
            return np.nan

        def map_a40(val):
            if pd.isna(val):
                return np.nan
            if val == 'Cooperative':
                return 1
            if val == 'Uncooperative':
                return 0
            return np.nan

        def map_a41(val):
            if pd.isna(val):
                return np.nan
            if val in ['Before_Investigation', 'During_Investigation']:
                return 1
            if val == 'None':
                return 0
            return np.nan

        field_mappers.update({
            'A27_DamageEstablished': map_a27,
            'A28_DPIARequired': map_a28,
            'A29_SecurityMeasures': map_a29,
            'A30_DataMinimization': map_compliance,
            'A33_TransparencyObligations': map_compliance,
            'A40_DefendantCooperation': map_a40,
            'A41_MitigatingMeasures': map_a41,
        })

        created = 0
        for col, mapper in field_mappers.items():
            if col in df_clean.columns:
                out_col = f"{col}_pos_bin"
                df_clean[out_col] = df_clean[col].apply(mapper)
                created += 1

        print(f"  Created {created} _pos_bin columns")
        self.log_action("tri_state_pos_bin", f"Created {created} pos bins", created)

        return df_clean

    def one_hot_single_select_enums(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot expand single-select enums using fixed vocab, including UNKNOWN/N_A where defined."""
        print("🔧 One-hot expanding single-select enums...")

        df_clean = df.copy()

        # Import vocab from schema where available
        try:
            from gdpr_schema_validation import (
                CASE_TRIGGERS, DPA_ROLES, APPEAL_SUCCESS, PRIOR_INFRINGEMENTS,
                DEFENDANT_ROLES, DEFENDANT_CATEGORIES, DAMAGE_TYPES, DPIA_STATUS,
                SECURITY_STATUS, COMPLIANCE_STATUS, YES_NO_UNKNOWN
            )
        except Exception:
            CASE_TRIGGERS = []
            DPA_ROLES = []
            APPEAL_SUCCESS = []
            PRIOR_INFRINGEMENTS = []
            DEFENDANT_ROLES = []
            DEFENDANT_CATEGORIES = []
            DAMAGE_TYPES = []
            DPIA_STATUS = []
            SECURITY_STATUS = []
            COMPLIANCE_STATUS = []
            YES_NO_UNKNOWN = []

        enum_map: Dict[str, List[str]] = {
            'A4_CaseTrigger': CASE_TRIGGERS,
            'A6_DPARole': DPA_ROLES,
            'A8_AppealSuccess': APPEAL_SUCCESS,
            'A9_PriorInfringements': PRIOR_INFRINGEMENTS,
            'A12_DefendantRole': DEFENDANT_ROLES,
            'A15_DefendantCategory': DEFENDANT_CATEGORIES,
            'A27_DamageEstablished': DAMAGE_TYPES,
            'A28_DPIARequired': DPIA_STATUS,
            'A29_SecurityMeasures': SECURITY_STATUS,
            'A30_DataMinimization': COMPLIANCE_STATUS,
            'A33_TransparencyObligations': COMPLIANCE_STATUS,
            'A37_LegalBasisInvalid': ['Y', 'N', 'Not_Evaluated', 'UNKNOWN'],
            'A38_NegligenceEstablished': ['Y', 'N', 'Not_Assessed', 'UNKNOWN'],
            'A39_ManagementAuthorization': ['Y', 'N', 'Not_Assessed', 'UNKNOWN'],
            'A40_DefendantCooperation': ['Cooperative', 'Partially_Cooperative', 'Uncooperative', 'Not_Discussed', 'UNKNOWN'],
            'A41_MitigatingMeasures': ['Before_Investigation', 'During_Investigation', 'None', 'Not_Discussed', 'UNKNOWN'],
            'A42_PriorNonCompliance': ['Y', 'N', 'Not_Assessed', 'UNKNOWN'],
            'A43_FinancialBenefit': ['Y', 'N', 'Not_Assessed', 'UNKNOWN'],
            'A49_FineCalculationFactors': ['Y', 'Partially', 'N', 'N_A']
        }

        total_created = 0
        for col, allowed in enum_map.items():
            if col not in df_clean.columns or not allowed:
                continue
            for v in allowed:
                out_col = f"{col}__{v}"
                df_clean[out_col] = (df_clean[col] == v).astype(int)
                total_created += 1

        print(f"  One-hot columns created: {total_created}")
        self.log_action("one_hot_enums", f"Created {total_created} enum dummies", total_created)

        return df_clean

    def split_multi_select_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split multi-select fields for easier analysis."""
        print("🔧 Processing multi-select fields...")

        df_clean = df.copy()
        multi_select_fields = {
            'A17_SensitiveDataTypes': ('SensitiveType_', SENSITIVE_DATA_TYPES),
            'A19_VulnerableSubjects': ('VulnerableSubject_', VULNERABLE_SUBJECTS),
            'A20_LegalBasis': ('LegalBasis_', LEGAL_BASIS),
            'A22_TransferMechanism': ('TransferMech_', TRANSFER_MECHANISMS),
            'A32_RightsInvolved': ('Right_', RIGHTS_TYPES),
            'A45_SanctionType': ('Sanction_', SANCTION_TYPES)
        }

        exclude_tokens = {'N_A', 'UNKNOWN', 'None_Mentioned', 'Not_Specified'}

        for field, (prefix, allowed_values) in multi_select_fields.items():
            if field in df_clean.columns:
                # Create binary columns for the fixed vocabulary (excluding meta tokens)
                fixed_values = [v for v in allowed_values if v not in exclude_tokens]
                created = 0
                for value in fixed_values:
                    safe_value = str(value).strip()
                    if not safe_value:
                        continue
                    col_name = f"{prefix}{safe_value.replace(' ', '_').replace('(', '').replace(')', '')}"
                    if col_name not in df_clean.columns:
                        df_clean[col_name] = 0
                    created += 1

                # Populate indicators from the cell contents
                for idx, cell_value in df_clean[field].items():
                    if pd.notna(cell_value) and cell_value not in exclude_tokens:
                        cv = str(cell_value)
                        if cv.startswith('[') and cv.endswith(']'):
                            cv = cv[1:-1]
                        values = [v.strip() for v in cv.split(',') if v.strip()]
                        for v in values:
                            if v in fixed_values:
                                col_name = f"{prefix}{v.replace(' ', '_').replace('(', '').replace(')', '')}"
                                df_clean.loc[idx, col_name] = 1

                print(f"  {field}: Created {created} binary indicators (fixed vocab)")
                self.log_action("split_multi_select", f"{field}: {created} indicators (fixed)", created)

        return df_clean

    def canonicalize_column_order(self, original_columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """Apply canonical column ordering for cross-file consistency."""
        df_clean = df.copy()

        # Prefer schema-defined base order when available
        try:
            from gdpr_schema_validation import gdpr_schema
            base_cols = list(gdpr_schema.columns.keys())  # type: ignore[attr-defined]
        except Exception:
            base_cols = list(original_columns)
        all_cols = set(df_clean.columns)

        eur_cols = ['A46_FineAmount_EUR'] if 'A46_FineAmount_EUR' in all_cols else []

        a35_cols = sorted([c for c in all_cols if c.startswith('A35_Art_')],
                          key=lambda c: int(re.findall(r'\d+', c)[0]) if re.findall(r'\d+', c) else 0)

        prefixes = ['SensitiveType_', 'VulnerableSubject_', 'LegalBasis_', 'TransferMech_', 'Right_', 'Sanction_']
        by_prefix = []
        for p in prefixes:
            cols = sorted([c for c in all_cols if c.startswith(p)])
            by_prefix.extend(cols)

        bin_cols = sorted([c for c in all_cols if c.endswith('_bin')])

        used = set(base_cols + eur_cols + a35_cols + by_prefix + bin_cols)
        other_cols = sorted([c for c in all_cols if c not in used])

        final_order = base_cols + eur_cols + a35_cols + by_prefix + bin_cols + other_cols

        # Ensure all columns exist
        for col in final_order:
            if col not in df_clean.columns:
                df_clean[col] = np.nan

        return df_clean[final_order]

    def validate_logical_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix logical inconsistencies between related fields."""
        print("🔧 Validating logical consistency...")

        df_clean = df.copy()
        fixes = 0

        # Fix fine amount vs sanction type consistency
        if 'A46_FineAmount' in df_clean.columns and 'A45_SanctionType' in df_clean.columns:
            for idx, row in df_clean.iterrows():
                fine_amount = row['A46_FineAmount']
                sanction_type = row['A45_SanctionType']

                try:
                    amount = float(fine_amount) if pd.notna(fine_amount) else 0
                    has_fine = amount > 0
                    mentions_fine = 'Fine' in str(sanction_type) if pd.notna(sanction_type) else False

                    # If fine amount > 0 but sanction doesn't mention Fine, add it
                    if has_fine and not mentions_fine and pd.notna(sanction_type) and sanction_type != 'N_A':
                        if sanction_type == 'None':
                            df_clean.loc[idx, 'A45_SanctionType'] = 'Fine'
                        else:
                            df_clean.loc[idx, 'A45_SanctionType'] = f"{sanction_type}, Fine"
                        fixes += 1

                except (ValueError, TypeError):
                    continue

        print(f"  Fixed {fixes} logical inconsistencies")
        self.log_action("validate_consistency", f"Fixed {fixes} logical inconsistencies", fixes)

        return df_clean

    def remove_low_quality_rows(self, df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """Remove rows with very low data quality."""
        print(f"🔧 Removing rows with <{threshold*100}% data quality...")

        df_clean = df.copy()

        # Calculate quality score for each row
        quality_scores = []
        for idx, row in df_clean.iterrows():
            non_missing = 0
            total_fields = 0

            for col in df_clean.columns:
                if col not in ['ID']:  # Exclude ID from quality calculation
                    total_fields += 1
                    if pd.notna(row[col]) and row[col] not in ['UNKNOWN', 'N_A', '']:
                        non_missing += 1

            quality_score = non_missing / total_fields if total_fields > 0 else 0
            quality_scores.append(quality_score)

        # Identify low quality rows
        low_quality_mask = np.array(quality_scores) < threshold
        low_quality_count = low_quality_mask.sum()

        if low_quality_count > 0:
            print(f"  Identified {low_quality_count} low-quality rows for removal")

            # Log which rows are being removed
            removed_ids = df_clean.loc[low_quality_mask, 'ID'].tolist() if 'ID' in df_clean.columns else list(df_clean.index[low_quality_mask])
            print(f"  Removing rows: {removed_ids}")

            df_clean = df_clean.loc[~low_quality_mask].reset_index(drop=True)

        self.log_action("remove_low_quality", f"Removed {low_quality_count} low-quality rows", low_quality_count)

        return df_clean

    def generate_cleaning_statistics(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        """Generate statistics about the cleaning process."""
        self.statistics = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'rows_removed': len(original_df) - len(cleaned_df),
            'original_columns': len(original_df.columns),
            'cleaned_columns': len(cleaned_df.columns),
            'columns_added': len(cleaned_df.columns) - len(original_df.columns),
            'total_cleaning_actions': len(self.cleaning_log),
            'original_missing_values': original_df.isna().sum().sum(),
            'cleaned_missing_values': cleaned_df.isna().sum().sum(),
            'missing_values_handled': original_df.isna().sum().sum() - cleaned_df.isna().sum().sum()
        }

    def clean_dataset(self, df: pd.DataFrame, remove_low_quality: bool = True, quality_threshold: float = 0.3) -> pd.DataFrame:
        """Run the complete cleaning pipeline."""
        print("🚀 Starting Enhanced Data Cleaning Pipeline")
        print("=" * 60)

        original_df = df.copy()
        cleaned_df = df.copy()

        # Step 1: Remove bracket inconsistencies
        cleaned_df = self.remove_brackets(cleaned_df)

        # Step 2: Standardize missing values
        cleaned_df = self.standardize_missing_values(cleaned_df)

        # Step 3: Clean dates
        cleaned_df = self.clean_dates(cleaned_df)

        # Step 4: Normalize currency
        cleaned_df = self.normalize_currency(cleaned_df)

        # Step 5: Clean GDPR articles
        cleaned_df = self.clean_gdpr_articles(cleaned_df)
        cleaned_df = self.expand_a34_top_level_articles(cleaned_df)

        # Step 6: Expand A35 top-level article indicators (1 for present, NaN otherwise)
        cleaned_df = self.expand_a35_top_level_articles(cleaned_df)

        # Step 7: Convert Y/N style fields to binary indicators
        cleaned_df = self.convert_yes_no_fields_to_binary(cleaned_df)

        # Step 7b: Create tri-state positive bins
        cleaned_df = self.convert_tri_state_fields_to_pos_bin(cleaned_df)

        # Step 7c: One-hot selected single-select enums with fixed vocab
        cleaned_df = self.one_hot_single_select_enums(cleaned_df)

        # Step 8: Split multi-select fields (fixed vocab) after normalizing sanction tokens
        cleaned_df = self.normalize_sanction_type(cleaned_df)
        cleaned_df = self.split_multi_select_fields(cleaned_df)

        # Step 9: Validate logical consistency
        cleaned_df = self.validate_logical_consistency(cleaned_df)

        # Step 10: Remove low quality rows (optional)
        if remove_low_quality:
            cleaned_df = self.remove_low_quality_rows(cleaned_df, quality_threshold)

        # Step 11: Canonicalize column order for cross-file consistency
        cleaned_df = self.canonicalize_column_order(original_df.columns.tolist(), cleaned_df)

        # Add dataset source column
        if 'dataset_source' not in cleaned_df.columns:
            cleaned_df.insert(0, 'dataset_source', '')

        # Generate statistics
        self.generate_cleaning_statistics(original_df, cleaned_df)

        print(f"\n✅ Cleaning pipeline completed!")
        print(f"  Original: {self.statistics['original_rows']} rows, {self.statistics['original_columns']} columns")
        print(f"  Cleaned:  {self.statistics['cleaned_rows']} rows, {self.statistics['cleaned_columns']} columns")
        print(f"  Actions:  {self.statistics['total_cleaning_actions']} cleaning operations performed")

        return cleaned_df

    def save_cleaning_report(self, output_file: str):
        """Save detailed cleaning report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.statistics,
            'cleaning_log': self.cleaning_log,
            'currency_rates_used': self.currency_rates
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📁 Cleaning report saved to: {output_file}")

def main():
    """Main execution function."""
    print("Enhanced GDPR Data Cleaning Pipeline")
    print("=" * 60)

    # CLI arguments
    parser = argparse.ArgumentParser(description="Enhanced GDPR Data Cleaning Pipeline")
    parser.add_argument('--input', '-i', dest='input_path', default='dataNorway.csv', help='Path to input CSV file')
    parser.add_argument('--quality-threshold', '-q', dest='quality_threshold', type=float, default=0.3,
                        help='Row quality threshold (0-1). Rows below are removed if enabled')
    parser.add_argument('--no-remove-low-quality', dest='no_remove', action='store_true',
                        help='Disable removal of low-quality rows')
    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.input_path)
        print(f"✓ Loaded data from {args.input_path}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data '{args.input_path}': {e}")
        return

    # Initialize cleaner
    cleaner = GDPRDataCleaner()

    # Clean data
    remove_lq = not args.no_remove
    cleaned_df = cleaner.clean_dataset(df, remove_low_quality=remove_lq, quality_threshold=args.quality_threshold)

    # Save cleaned data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = Path(args.input_path).stem
    cleaned_file = f'{base_name}_cleaned_{timestamp}.csv'
    # Populate dataset_source with input filename
    try:
        cleaned_df['dataset_source'] = Path(args.input_path).name
    except Exception:
        cleaned_df['dataset_source'] = str(args.input_path)
    cleaned_df.to_csv(cleaned_file, index=False)
    print(f"📁 Cleaned data saved to: {cleaned_file}")

    # Save cleaning report
    report_file = f'data_cleaning_report_{timestamp}.json'
    cleaner.save_cleaning_report(report_file)

    print(f"\n🎯 Summary:")
    print(f"  - Data quality improved through {len(cleaner.cleaning_log)} operations")
    print(f"  - {cleaner.statistics['missing_values_handled']} missing values handled")
    print(f"  - {cleaner.statistics['columns_added']} new analytical columns created")
    print(f"  - Dataset ready for multi-format export and analysis")

if __name__ == "__main__":
    main()