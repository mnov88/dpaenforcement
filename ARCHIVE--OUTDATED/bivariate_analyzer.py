#!/usr/bin/env python3
"""
Enhanced Bivariate Analysis Engine for GDPR Enforcement Data
===========================================================

This module provides comprehensive bivariate statistical analysis capabilities
for exploring relationships between variables in GDPR enforcement datasets.

Features:
- Categorical-Categorical: Chi-square tests, Cramér's V, contingency analysis
- Continuous-Categorical: ANOVA, Kruskal-Wallis, effect size measures
- Continuous-Continuous: Correlation analysis with significance testing
- Interactive visualizations and comprehensive reporting
- Business intelligence focused on enforcement patterns

Author: Enhanced GDPR Data Analysis Pipeline
Version: 2.0
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Optional, Any
import argparse

# Statistical libraries
# from scipy import stats  # not used directly
from scipy.stats import chi2_contingency, fisher_exact, kruskal, f_oneway, pearsonr, spearmanr
import itertools

# Visualization libraries
# Plotly imports not used in this module's current scope
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.offline as pyo

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BivariateAnalyzer:
    """
    Comprehensive bivariate analysis engine for GDPR enforcement data.

    This class provides statistical relationship analysis between pairs of variables,
    including categorical-categorical, continuous-categorical, and continuous-continuous
    relationships with appropriate statistical tests and effect size measures.
    """

    def __init__(self, data_path: str, univariate_results_path: Optional[str] = None):
        """
        Initialize the BivaviateAnalyzer with data and optional univariate results.

        Args:
            data_path: Path to the cleaned dataset
            univariate_results_path: Optional path to univariate analysis results
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load data
        self.df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")

        # Load univariate results if provided
        self.univariate_results = None
        if univariate_results_path and Path(univariate_results_path).exists():
            with open(univariate_results_path, 'r') as f:
                univariate_data = json.load(f)
                # Handle nested structure
                if 'complete' in univariate_data:
                    self.univariate_results = univariate_data['complete']
                else:
                    self.univariate_results = univariate_data
            self.logger.info("Loaded univariate analysis results")

        # Initialize analysis components
        self.variable_types = {}
        self.bivariate_results = {
            'categorical_categorical': {},
            'continuous_categorical': {},
            'continuous_continuous': {},
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_cases': len(self.df),
                'data_source': data_path
            }
        }

        # Classify variables
        self._classify_variables()

    def _classify_variables(self):
        """Classify variables into categorical, continuous, binary, temporal types."""
        self.logger.info("Classifying variables for bivariate analysis...")

        # Use univariate results if available, otherwise classify from scratch
        if self.univariate_results:
            # Map from univariate analysis structure
            for category_type in ['categorical', 'continuous', 'binary', 'temporal']:
                if category_type in self.univariate_results:
                    for var in self.univariate_results[category_type].keys():
                        if var in self.df.columns:
                            self.variable_types[var] = category_type

        # Fallback classification for any missing variables
        for col in self.df.columns:
            if col == 'ID' or col in self.variable_types:
                continue

            # Continuous variables
            if col in ['A46_FineAmount', 'A46_FineAmount_EUR'] and pd.api.types.is_numeric_dtype(self.df[col]):
                self.variable_types[col] = 'continuous'
            # Date variables
            elif col in ['A3_DecisionDate']:
                self.variable_types[col] = 'temporal'
            # Binary variables (including transformed ones)
            elif (col.startswith('SensitiveType_') or col.startswith('VulnerableSubject_') or
                  col.startswith('LegalBasis_') or col.startswith('TransferMech_') or
                  col.startswith('Right_') or col.startswith('Sanction_') or
                  col in ['A5_CrossBorder', 'A7_IsAppeal', 'A16_SensitiveData', 'A18_Cookies',
                         'A21_DataTransfers', 'A31_SubjectRightsRequests', 'A36_NoInfringement', 'A48_EDPBGuidelines']):
                self.variable_types[col] = 'binary'
            # Categorical variables
            else:
                self.variable_types[col] = 'categorical'

        # Log classification summary
        type_counts = {}
        for var_type in self.variable_types.values():
            type_counts[var_type] = type_counts.get(var_type, 0) + 1

        self.logger.info("Variable classification complete:")
        for var_type, count in type_counts.items():
            self.logger.info(f"  {var_type.title()}: {count} variables")

    def _cramers_v(self, confusion_matrix: np.ndarray) -> float:
        """Calculate Cramér's V effect size for categorical associations."""
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for mean differences."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def analyze_categorical_categorical(self, var1: str, var2: str) -> Dict[str, Any]:
        """
        Analyze relationship between two categorical variables.

        Args:
            var1, var2: Names of categorical variables

        Returns:
            Dictionary with statistical test results and effect sizes
        """
        # Create contingency table
        contingency = pd.crosstab(self.df[var1], self.df[var2], dropna=False)

        # Choose appropriate test based on table size/sparsity
        n_total = contingency.sum().sum()
        test_used = 'chi_square'
        chi2 = np.nan
        p_value = np.nan
        dof = None
        expected = None

        if contingency.shape == (2, 2):
            try:
                # Fisher's exact test for 2x2 tables
                _, p_value = fisher_exact(contingency.values)
                test_used = 'fisher_exact'
            except Exception:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                test_used = 'chi_square'
        else:
            chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Effect size (Cramér's V)
        cramers_v = self._cramers_v(contingency.values)

        # Validity diagnostics where expected available
        if expected is not None:
            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()
            total_cells = expected.size
            valid_flag = (min_expected >= 1) and (cells_below_5 / total_cells <= 0.2)
        else:
            min_expected = None
            cells_below_5 = None
            total_cells = int(contingency.size)
            valid_flag = True

        return {
            'var1': var1,
            'var2': var2,
            'test_type': test_used,
            'chi2_statistic': float(chi2) if not np.isnan(chi2) else None,
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof) if dof is not None else None,
            'cramers_v': float(cramers_v),
            'sample_size': int(n_total),
            'contingency_table': contingency.to_dict(),
            'test_validity': {
                'min_expected_frequency': float(min_expected) if min_expected is not None else None,
                'cells_below_5': int(cells_below_5) if cells_below_5 is not None else None,
                'total_cells': int(total_cells),
                'valid': valid_flag
            },
            'interpretation': self._interpret_categorical_relationship(cramers_v, p_value)
        }

    def analyze_continuous_categorical(self, continuous_var: str, categorical_var: str) -> Dict[str, Any]:
        """
        Analyze relationship between continuous and categorical variables.

        Args:
            continuous_var: Name of continuous variable
            categorical_var: Name of categorical variable

        Returns:
            Dictionary with statistical test results and effect sizes
        """
        # Remove missing values
        data = self.df[[continuous_var, categorical_var]].dropna()

        if len(data) < 3:
            return self._empty_result(continuous_var, categorical_var, 'insufficient_data')

        # Group continuous variable by categories
        groups = [group[continuous_var].values for name, group in data.groupby(categorical_var)]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups

        if len(groups) < 2:
            return self._empty_result(continuous_var, categorical_var, 'insufficient_groups')

        # Kruskal-Wallis test (non-parametric)
        kw_statistic, kw_p_value = kruskal(*groups)

        # ANOVA (parametric, for comparison)
        try:
            f_statistic, anova_p_value = f_oneway(*groups)
        except Exception:
            f_statistic, anova_p_value = np.nan, np.nan

        # Effect size calculation (eta-squared approximation)
        total_n = len(data)
        k = len(groups)
        eta_squared = (kw_statistic - k + 1) / (total_n - k) if total_n > k else 0
        eta_squared = max(0, min(1, eta_squared))  # Bound between 0 and 1

        # Group statistics
        group_stats = {}
        for name, group in data.groupby(categorical_var):
            values = group[continuous_var].values
            if len(values) > 0:
                group_stats[str(name)] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return {
            'continuous_var': continuous_var,
            'categorical_var': categorical_var,
            'test_type': 'kruskal_wallis',
            'kw_statistic': float(kw_statistic),
            'kw_p_value': float(kw_p_value),
            'anova_f_statistic': float(f_statistic) if not np.isnan(f_statistic) else None,
            'anova_p_value': float(anova_p_value) if not np.isnan(anova_p_value) else None,
            'eta_squared': float(eta_squared),
            'sample_size': int(total_n),
            'group_count': len(groups),
            'group_statistics': group_stats,
            'interpretation': self._interpret_continuous_categorical_relationship(eta_squared, kw_p_value)
        }

    def analyze_continuous_continuous(self, var1: str, var2: str) -> Dict[str, Any]:
        """
        Analyze relationship between two continuous variables.

        Args:
            var1, var2: Names of continuous variables

        Returns:
            Dictionary with correlation analysis results
        """
        # Remove missing values
        data = self.df[[var1, var2]].dropna()

        if len(data) < 3:
            return self._empty_result(var1, var2, 'insufficient_data')

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(data[var1], data[var2])

        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = spearmanr(data[var1], data[var2])

        # Sample size
        n = len(data)

        return {
            'var1': var1,
            'var2': var2,
            'test_type': 'correlation',
            'pearson_r': float(pearson_r),
            'pearson_p_value': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p_value': float(spearman_p),
            'sample_size': int(n),
            'interpretation': self._interpret_correlation(pearson_r, pearson_p, spearman_r, spearman_p)
        }

    def _empty_result(self, var1: str, var2: str, reason: str) -> Dict[str, Any]:
        """Return empty result structure for failed analyses."""
        return {
            'var1': var1,
            'var2': var2,
            'test_type': 'failed',
            'reason': reason,
            'interpretation': f"Analysis failed: {reason}"
        }

    def _interpret_categorical_relationship(self, cramers_v: float, p_value: float) -> str:
        """Interpret categorical relationship strength."""
        if p_value > 0.05:
            return "No significant association"
        elif cramers_v < 0.1:
            return "Weak association"
        elif cramers_v < 0.3:
            return "Moderate association"
        elif cramers_v < 0.5:
            return "Strong association"
        else:
            return "Very strong association"

    def _interpret_continuous_categorical_relationship(self, eta_squared: float, p_value: float) -> str:
        """Interpret continuous-categorical relationship strength."""
        if p_value > 0.05:
            return "No significant group differences"
        elif eta_squared < 0.01:
            return "Small effect"
        elif eta_squared < 0.06:
            return "Medium effect"
        elif eta_squared < 0.14:
            return "Large effect"
        else:
            return "Very large effect"

    def _interpret_correlation(self, pearson_r: float, pearson_p: float,
                              spearman_r: float, spearman_p: float) -> str:
        """Interpret correlation strength."""
        # Use Spearman if significantly different from Pearson (suggests non-linearity)
        primary_r = spearman_r if abs(pearson_r - spearman_r) > 0.1 else pearson_r
        primary_p = spearman_p if abs(pearson_r - spearman_r) > 0.1 else pearson_p

        if primary_p > 0.05:
            return "No significant correlation"
        elif abs(primary_r) < 0.1:
            return "Negligible correlation"
        elif abs(primary_r) < 0.3:
            return "Weak correlation"
        elif abs(primary_r) < 0.5:
            return "Moderate correlation"
        elif abs(primary_r) < 0.7:
            return "Strong correlation"
        else:
            return "Very strong correlation"

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive bivariate analysis on all appropriate variable pairs.

        Returns:
            Complete bivariate analysis results
        """
        self.logger.info("Starting comprehensive bivariate analysis...")

        # Get variables by type
        categorical_vars = [var for var, vtype in self.variable_types.items()
                          if vtype in ['categorical', 'binary'] and var in self.df.columns]
        continuous_vars = [var for var, vtype in self.variable_types.items()
                         if vtype == 'continuous' and var in self.df.columns]

        self.logger.info(f"Analyzing {len(categorical_vars)} categorical and {len(continuous_vars)} continuous variables")

        # Categorical-Categorical analysis
        self.logger.info("Analyzing categorical-categorical relationships...")
        cat_pairs = list(itertools.combinations(categorical_vars, 2))
        for var1, var2 in cat_pairs[:50]:  # Limit to prevent excessive computation
            try:
                result = self.analyze_categorical_categorical(var1, var2)
                pair_key = f"{var1}_vs_{var2}"
                self.bivariate_results['categorical_categorical'][pair_key] = result
            except Exception as e:
                self.logger.warning(f"Failed categorical analysis for {var1} vs {var2}: {e}")

        # Continuous-Categorical analysis
        self.logger.info("Analyzing continuous-categorical relationships...")
        for cont_var in continuous_vars:
            for cat_var in categorical_vars[:20]:  # Limit categorical variables
                try:
                    result = self.analyze_continuous_categorical(cont_var, cat_var)
                    pair_key = f"{cont_var}_vs_{cat_var}"
                    self.bivariate_results['continuous_categorical'][pair_key] = result
                except Exception as e:
                    self.logger.warning(f"Failed continuous-categorical analysis for {cont_var} vs {cat_var}: {e}")

        # Continuous-Continuous analysis
        self.logger.info("Analyzing continuous-continuous relationships...")
        cont_pairs = list(itertools.combinations(continuous_vars, 2))
        for var1, var2 in cont_pairs:
            try:
                result = self.analyze_continuous_continuous(var1, var2)
                pair_key = f"{var1}_vs_{var2}"
                self.bivariate_results['continuous_continuous'][pair_key] = result
            except Exception as e:
                self.logger.warning(f"Failed continuous analysis for {var1} vs {var2}: {e}")

        # Apply FDR corrections and generate summary statistics
        self._apply_fdr_corrections()
        self._generate_summary()

        self.logger.info("Comprehensive bivariate analysis complete")
        return self.bivariate_results

    def _generate_summary(self):
        """Generate summary statistics for the bivariate analysis."""
        summary = {
            'total_relationships_analyzed': (
                len(self.bivariate_results['categorical_categorical']) +
                len(self.bivariate_results['continuous_categorical']) +
                len(self.bivariate_results['continuous_continuous'])
            ),
            'categorical_categorical_count': len(self.bivariate_results['categorical_categorical']),
            'continuous_categorical_count': len(self.bivariate_results['continuous_categorical']),
            'continuous_continuous_count': len(self.bivariate_results['continuous_continuous']),
            'significant_relationships': 0,
            'strong_relationships': 0
        }

        # Count significant relationships
        for category in ['categorical_categorical', 'continuous_categorical', 'continuous_continuous']:
            for pair_key, result in self.bivariate_results[category].items():
                if 'p_value' in result and result.get('p_value', 1) < 0.05:
                    summary['significant_relationships'] += 1

                interpretation = result.get('interpretation', '')
                if 'strong' in interpretation.lower() or 'large' in interpretation.lower():
                    summary['strong_relationships'] += 1

        self.bivariate_results['summary'] = summary

    def _apply_fdr_corrections(self):
        """Apply Benjamini-Hochberg FDR within each family of tests."""
        try:
            from statsmodels.stats.multitest import multipletests
        except Exception:
            # If statsmodels not installed, skip
            return

        def correct_family(results_dict: Dict[str, Any], p_key: str, q_key: str = 'q_value'):
            items = list(results_dict.items())
            pvals = []
            idxs = []
            for i, (k, v) in enumerate(items):
                p = v.get(p_key)
                if p is not None:
                    pvals.append(p)
                    idxs.append(i)
            if not pvals:
                return
            reject, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            for flag, q, i in zip(reject, qvals, idxs):
                key, val = items[i]
                val[q_key] = float(q)
                val['significant_fdr_05'] = bool(flag)
                results_dict[key] = val

        correct_family(self.bivariate_results['categorical_categorical'], p_key='p_value')
        correct_family(self.bivariate_results['continuous_categorical'], p_key='kw_p_value')
        correct_family(self.bivariate_results['continuous_continuous'], p_key='pearson_p_value')

    def save_results(self, output_path: str = None) -> str:
        """
        Save bivariate analysis results to JSON file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to saved results file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"bivariate_analysis_results_{timestamp}.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        converted_results = convert_types(self.bivariate_results)

        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2)

        self.logger.info(f"Bivariate analysis results saved to: {output_path}")
        return output_path

def main():
    """Main execution function."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Enhanced Bivariate Analysis Engine")
    print("=" * 50)

    # CLI args
    parser = argparse.ArgumentParser(description="Enhanced Bivariate Analysis Engine")
    parser.add_argument('--data', '-d', dest='data_path', help='Path to cleaned CSV; defaults to latest dataNorway_cleaned_*.csv')
    parser.add_argument('--univariate', '-u', dest='univariate_path', help='Path to univariate JSON; defaults to latest univariate_analysis_results_*.json')
    args = parser.parse_args()

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_files = list(Path('.').glob('dataNorway_cleaned_*.csv'))
        if not data_files:
            print("❌ No cleaned data files found. Please run enhanced_data_cleaning.py first.")
            sys.exit(1)
        data_path = str(sorted(data_files)[-1])
    print(f"📂 Loading data from: {data_path}")

    # Determine univariate path
    if args.univariate_path:
        univariate_path = args.univariate_path
    else:
        univariate_files = list(Path('.').glob('univariate_analysis_results_*.json'))
        univariate_path = str(sorted(univariate_files)[-1]) if univariate_files else None

    if univariate_path:
        print(f"📊 Loading univariate results from: {univariate_path}")

    # Initialize analyzer
    analyzer = BivariateAnalyzer(data_path, univariate_path)

    # Run comprehensive analysis
    print("🚀 Running comprehensive bivariate analysis...")
    results = analyzer.run_comprehensive_analysis()

    # Save results
    output_path = analyzer.save_results()

    print("\n✅ Bivariate Analysis Complete!")
    print(f"📁 Results saved to: {output_path}")
    print("📊 Relationships analyzed:")
    print(f"  Categorical-Categorical: {results['summary']['categorical_categorical_count']}")
    print(f"  Continuous-Categorical: {results['summary']['continuous_categorical_count']}")
    print(f"  Continuous-Continuous: {results['summary']['continuous_continuous_count']}")
    print(f"🔍 Significant relationships found: {results['summary']['significant_relationships']}")
    print(f"💪 Strong relationships found: {results['summary']['strong_relationships']}")

if __name__ == "__main__":
    main()