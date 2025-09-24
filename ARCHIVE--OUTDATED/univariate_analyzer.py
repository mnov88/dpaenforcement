#!/usr/bin/env python3
"""
Enhanced Univariate Analysis Engine
===================================

Comprehensive univariate analysis system for GDPR enforcement data with
intelligent variable classification, automated insights, and detailed logging.

Author: Enhanced Data Analysis Pipeline
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import entropy, normaltest, shapiro, jarque_bera
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('univariate_analysis.log'),
        logging.StreamHandler()
    ]
)

class UnivariateAnalyzer:
    """Comprehensive univariate analysis engine with intelligent insights."""

    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer with dataset."""
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)
        self.analysis_results = {}
        self.insights = []
        self.variable_profiles = {}

        # Analysis parameters
        self.rare_threshold = 0.05  # 5% threshold for rare categories
        self.outlier_threshold = 3  # Z-score threshold
        self.min_binary_samples = 5  # Minimum samples for binary analysis

        self.logger.info(f"Initialized UnivariateAnalyzer with {len(df)} rows, {len(df.columns)} columns")

        # Classify variables immediately
        self._classify_variables()

    def _classify_variables(self):
        """Intelligently classify variables by type and characteristics."""
        self.logger.info("Starting intelligent variable classification...")

        self.categorical_vars = []
        self.continuous_vars = []
        self.binary_vars = []
        self.date_vars = []
        self.text_vars = []
        self.id_vars = []

        for col in self.df.columns:
            # Skip ID columns
            if col.lower() in ['id', 'index'] or col.startswith('ID'):
                self.id_vars.append(col)
                continue

            # Check for date columns
            if 'date' in col.lower() or col in ['A3_DecisionDate']:
                self.date_vars.append(col)
                continue

            # Get basic statistics
            unique_count = self.df[col].nunique()
            non_null_count = self.df[col].notna().sum()
            total_count = len(self.df)

            # Text variables (high cardinality, mostly unique)
            if (col in ['A11_DefendantName', 'A50_CaseSummary'] or
                unique_count > 0.8 * non_null_count):
                self.text_vars.append(col)

            # Binary variables
            elif (
                (
                    (str(self.df[col].dtype) in ['int64', 'float64', 'bool']) and
                    len(set(self.df[col].dropna().unique())) > 0 and
                    set(pd.Series(self.df[col].dropna().astype(int)).unique()).issubset({0, 1})
                )
                or
                (
                    self.df[col].dtype == 'object' and
                    set(pd.Series(self.df[col].dropna().astype(str).str.upper()).unique()).issubset({'Y', 'N'})
                )
            ):
                self.binary_vars.append(col)

            # Continuous variables
            elif (self.df[col].dtype in ['int64', 'float64'] and
                  unique_count > 10 and unique_count > 0.1 * non_null_count):
                self.continuous_vars.append(col)

            # Categorical variables (everything else)
            else:
                self.categorical_vars.append(col)

        # Log classification results
        self.logger.info(f"Variable classification complete:")
        self.logger.info(f"  Categorical: {len(self.categorical_vars)} variables")
        self.logger.info(f"  Continuous: {len(self.continuous_vars)} variables")
        self.logger.info(f"  Binary: {len(self.binary_vars)} variables")
        self.logger.info(f"  Date: {len(self.date_vars)} variables")
        self.logger.info(f"  Text: {len(self.text_vars)} variables")
        self.logger.info(f"  ID: {len(self.id_vars)} variables")

        # Create variable profiles
        self._create_variable_profiles()

    def _create_variable_profiles(self):
        """Create detailed profiles for each variable."""
        self.logger.info("Creating detailed variable profiles...")

        for col in self.df.columns:
            if col in self.id_vars:
                continue

            profile = {
                'column_name': col,
                'data_type': str(self.df[col].dtype),
                'variable_type': self._get_variable_type(col),
                'total_count': len(self.df),
                'non_null_count': self.df[col].notna().sum(),
                'null_count': self.df[col].isna().sum(),
                'null_percentage': (self.df[col].isna().sum() / len(self.df)) * 100,
                'unique_count': self.df[col].nunique(),
                'uniqueness_ratio': self.df[col].nunique() / self.df[col].notna().sum() if self.df[col].notna().sum() > 0 else 0
            }

            # Add type-specific metrics
            if col in self.continuous_vars:
                profile.update(self._get_continuous_profile(col))
            elif col in self.categorical_vars:
                profile.update(self._get_categorical_profile(col))
            elif col in self.binary_vars:
                profile.update(self._get_binary_profile(col))

            self.variable_profiles[col] = profile

        self.logger.info(f"Created profiles for {len(self.variable_profiles)} variables")

    def _get_variable_type(self, col: str) -> str:
        """Get the classified type of a variable."""
        if col in self.categorical_vars:
            return 'categorical'
        elif col in self.continuous_vars:
            return 'continuous'
        elif col in self.binary_vars:
            return 'binary'
        elif col in self.date_vars:
            return 'date'
        elif col in self.text_vars:
            return 'text'
        else:
            return 'id'

    def _get_continuous_profile(self, col: str) -> Dict[str, Any]:
        """Get profile metrics for continuous variables."""
        data = self.df[col].dropna()
        if len(data) == 0:
            return {}

        return {
            'min_value': float(data.min()),
            'max_value': float(data.max()),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis()),
            'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else np.inf
        }

    def _get_categorical_profile(self, col: str) -> Dict[str, Any]:
        """Get profile metrics for categorical variables."""
        data = self.df[col].dropna()
        if len(data) == 0:
            return {}

        value_counts = data.value_counts()

        return {
            'most_frequent_value': value_counts.index[0],
            'most_frequent_count': int(value_counts.iloc[0]),
            'most_frequent_percentage': (value_counts.iloc[0] / len(data)) * 100,
            'entropy': entropy(value_counts.values),
            'rare_categories': sum(value_counts / len(data) < self.rare_threshold)
        }

    def _get_binary_profile(self, col: str) -> Dict[str, Any]:
        """Get profile metrics for binary variables."""
        data = self.df[col].dropna()
        if len(data) == 0:
            return {}

        # Determine positive value
        if data.dtype in ['int64', 'float64']:
            positive_val = 1
            positive_count = (data == 1).sum()
        else:
            positive_val = 'Y'
            positive_count = (data == 'Y').sum()

        return {
            'positive_value': positive_val,
            'positive_count': int(positive_count),
            'positive_percentage': (positive_count / len(data)) * 100,
            'negative_count': int(len(data) - positive_count),
            'negative_percentage': ((len(data) - positive_count) / len(data)) * 100
        }

    def analyze_categorical_variables(self) -> Dict[str, Any]:
        """Comprehensive analysis of categorical variables."""
        self.logger.info("Starting categorical variable analysis...")

        categorical_results = {}

        for col in self.categorical_vars:
            self.logger.info(f"Analyzing categorical variable: {col}")

            # Basic frequency analysis
            data = self.df[col].dropna()
            value_counts = data.value_counts()
            value_percentages = data.value_counts(normalize=True) * 100

            # Information content analysis
            info_entropy = entropy(value_counts.values)

            # Rare category analysis
            rare_categories = value_percentages[value_percentages < self.rare_threshold * 100]

            result = {
                'column_name': col,
                'total_observations': len(data),
                'unique_categories': len(value_counts),
                'entropy': info_entropy,
                'most_common_category': value_counts.index[0],
                'most_common_count': int(value_counts.iloc[0]),
                'most_common_percentage': float(value_percentages.iloc[0]),
                'rare_categories_count': len(rare_categories),
                'rare_categories': rare_categories.to_dict(),
                'frequency_distribution': value_counts.to_dict(),
                'percentage_distribution': value_percentages.to_dict()
            }

            # Add insights
            insights = []

            if len(rare_categories) > 0:
                insights.append(f"Contains {len(rare_categories)} rare categories (<{self.rare_threshold*100}%)")

            if info_entropy < 1:
                insights.append("Low diversity - heavily concentrated in few categories")
            elif info_entropy > 3:
                insights.append("High diversity - well distributed across categories")

            if value_percentages.iloc[0] > 80:
                insights.append(f"Heavily dominated by '{value_counts.index[0]}' ({value_percentages.iloc[0]:.1f}%)")

            result['insights'] = insights
            categorical_results[col] = result

        self.analysis_results['categorical'] = categorical_results
        self.logger.info(f"Completed categorical analysis for {len(categorical_results)} variables")

        return categorical_results

    def analyze_continuous_variables(self) -> Dict[str, Any]:
        """Comprehensive analysis of continuous variables."""
        self.logger.info("Starting continuous variable analysis...")

        continuous_results = {}

        for col in self.continuous_vars:
            self.logger.info(f"Analyzing continuous variable: {col}")

            data = self.df[col].dropna()
            if len(data) == 0:
                continue

            # Descriptive statistics
            desc_stats = {
                'count': len(data),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'mode': float(data.mode().iloc[0]) if len(data.mode()) > 0 else None,
                'std': float(data.std()),
                'variance': float(data.var()),
                'min': float(data.min()),
                'max': float(data.max()),
                'range': float(data.max() - data.min()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else np.inf
            }

            # Distribution fitting
            distribution_tests = self._test_distributions(data)

            # Outlier detection
            outliers = self._detect_outliers(data, col)

            # Percentile analysis
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentiles[f'p{p}'] = float(data.quantile(p/100))

            result = {
                'column_name': col,
                'descriptive_statistics': desc_stats,
                'distribution_tests': distribution_tests,
                'outliers': outliers,
                'percentiles': percentiles
            }

            # Generate insights
            insights = []

            if abs(desc_stats['skewness']) > 1:
                skew_type = "right" if desc_stats['skewness'] > 0 else "left"
                insights.append(f"Highly {skew_type}-skewed distribution")

            if desc_stats['coefficient_of_variation'] > 1:
                insights.append("High variability relative to mean")

            if len(outliers['extreme_outliers']) > 0:
                insights.append(f"Contains {len(outliers['extreme_outliers'])} extreme outliers")

            if distribution_tests['normality']['is_normal']:
                insights.append("Data appears to be normally distributed")
            else:
                insights.append("Data is not normally distributed")

            # Special insights for fine amounts
            if 'fine' in col.lower() or 'amount' in col.lower():
                if desc_stats['median'] < desc_stats['mean']:
                    insights.append("Few very high fines pulling the average up")

                zero_fines = (data == 0).sum()
                if zero_fines > 0:
                    insights.append(f"{zero_fines} cases with zero fines (compliance orders)")

            result['insights'] = insights
            continuous_results[col] = result

        self.analysis_results['continuous'] = continuous_results
        self.logger.info(f"Completed continuous analysis for {len(continuous_results)} variables")

        return continuous_results

    def _test_distributions(self, data: pd.Series) -> Dict[str, Any]:
        """Test data against common distributions."""
        results = {}

        # Normality tests
        if len(data) >= 8:  # Minimum sample size for Shapiro-Wilk
            try:
                shapiro_stat, shapiro_p = shapiro(data)
                results['normality'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            except:
                results['normality'] = {'test': 'Failed', 'is_normal': False}

        # Log-normal test (for positive values)
        if (data > 0).all():
            try:
                log_data = np.log(data)
                log_shapiro_stat, log_shapiro_p = shapiro(log_data)
                results['log_normality'] = {
                    'test': 'Log-Normal (Shapiro-Wilk on log)',
                    'statistic': float(log_shapiro_stat),
                    'p_value': float(log_shapiro_p),
                    'is_log_normal': log_shapiro_p > 0.05
                }
            except:
                results['log_normality'] = {'test': 'Failed', 'is_log_normal': False}

        return results

    def _detect_outliers(self, data: pd.Series, col_name: str) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        outlier_results = {}

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        moderate_outliers = data[z_scores > 2].tolist()
        extreme_outliers_z = data[z_scores > 3].tolist()

        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)].tolist()

        # Robust MAD-based method
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        if mad > 0:
            modified_z = 0.6745 * (data - median_val) / mad
            mad_outliers = data[np.abs(modified_z) > 3.5].tolist()
        else:
            mad_outliers = []

        # Isolation Forest (if enough data)
        if len(data) >= 10:
            try:
                # Set contamination dynamically from IQR outlier share within reasonable bounds
                approx_share = min(0.2, max(0.01, len(iqr_outliers) / max(1, len(data))))
                iso_forest = IsolationForest(contamination=approx_share, random_state=42)
                outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
                isolation_outliers = data[outlier_labels == -1].tolist()
            except:
                isolation_outliers = []
        else:
            isolation_outliers = []

        # Combine extreme outliers from multiple robust methods
        extreme_union = list(pd.Series(extreme_outliers_z + mad_outliers).dropna().unique())

        outlier_results = {
            'zscore_moderate': moderate_outliers,
            'zscore_extreme': extreme_outliers_z,
            'mad_outliers': mad_outliers,
            'iqr_outliers': iqr_outliers,
            'isolation_outliers': isolation_outliers,
            'extreme_outliers': extreme_union,
            'parameters': {
                'iqr': float(IQR) if pd.notna(IQR) else None,
                'median': float(median_val) if pd.notna(median_val) else None
            }
        }

        # Add case information for extreme outliers if this is fine data
        if 'fine' in col_name.lower() and len(extreme_union) > 0:
            outlier_cases = []
            for outlier_val in extreme_union:
                case_info = self.df[self.df[col_name] == outlier_val]
                if not case_info.empty:
                    case = case_info.iloc[0]
                    outlier_cases.append({
                        'value': outlier_val,
                        'case_id': case.get('ID', 'Unknown'),
                        'defendant': case.get('A11_DefendantName', 'Unknown'),
                        'country': case.get('A1_Country', 'Unknown')
                    })
            outlier_results['outlier_cases'] = outlier_cases

        return outlier_results

    def analyze_binary_variables(self) -> Dict[str, Any]:
        """Comprehensive analysis of binary variables."""
        self.logger.info("Starting binary variable analysis...")

        binary_results = {}

        for col in self.binary_vars:
            self.logger.info(f"Analyzing binary variable: {col}")

            data = self.df[col].dropna()
            if len(data) == 0:
                continue

            # Determine positive value and count
            if data.dtype in ['int64', 'float64']:
                positive_count = (data == 1).sum()
                negative_count = (data == 0).sum()
                positive_val = 1
            else:
                positive_count = (data == 'Y').sum()
                negative_count = (data == 'N').sum()
                positive_val = 'Y'

            total_count = len(data)
            positive_rate = positive_count / total_count

            # Confidence interval for proportion (Wilson score interval)
            confidence_interval = self._wilson_confidence_interval(positive_count, total_count)

            result = {
                'column_name': col,
                'total_observations': total_count,
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'positive_rate': float(positive_rate),
                'positive_percentage': float(positive_rate * 100),
                'confidence_interval_95': confidence_interval,
                'is_rare_event': positive_rate < 0.1 or positive_rate > 0.9
            }

            # Generate insights
            insights = []

            if positive_rate < 0.05:
                insights.append(f"Very rare positive event (<5%)")
            elif positive_rate > 0.95:
                insights.append(f"Very common positive event (>95%)")
            elif 0.4 <= positive_rate <= 0.6:
                insights.append("Well-balanced binary variable")

            if positive_count < 5 or negative_count < 5:
                insights.append("Low sample size - use caution in analysis")

            result['insights'] = insights
            binary_results[col] = result

        self.analysis_results['binary'] = binary_results
        self.logger.info(f"Completed binary analysis for {len(binary_results)} variables")

        return binary_results

    def _wilson_confidence_interval(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for a proportion."""
        if trials == 0:
            return (0.0, 0.0)

        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials

        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator

        return (max(0, centre - margin), min(1, centre + margin))

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in date variables."""
        self.logger.info("Starting temporal pattern analysis...")

        temporal_results = {}

        for col in self.date_vars:
            self.logger.info(f"Analyzing temporal variable: {col}")

            # Parse dates
            try:
                # Parse dates robustly, assuming day-first where ambiguous
                dates = pd.to_datetime(self.df[col], dayfirst=True, errors='coerce')
                valid_dates = dates.dropna()

                if len(valid_dates) == 0:
                    continue

                # Extract temporal components
                years = valid_dates.dt.year
                months = valid_dates.dt.month
                quarters = valid_dates.dt.quarter
                weekdays = valid_dates.dt.day_name()

                # Temporal distributions
                year_counts = years.value_counts().sort_index()
                month_counts = months.value_counts().sort_index()
                quarter_counts = quarters.value_counts().sort_index()
                weekday_counts = weekdays.value_counts()

                result = {
                    'column_name': col,
                    'total_dates': len(valid_dates),
                    'date_range': {
                        'earliest': valid_dates.min().strftime('%Y-%m-%d'),
                        'latest': valid_dates.max().strftime('%Y-%m-%d'),
                        'span_days': (valid_dates.max() - valid_dates.min()).days
                    },
                    'yearly_distribution': year_counts.to_dict(),
                    'monthly_distribution': month_counts.to_dict(),
                    'quarterly_distribution': quarter_counts.to_dict(),
                    'weekday_distribution': weekday_counts.to_dict()
                }

                # Generate insights
                insights = []

                # Year patterns
                if len(year_counts) > 1:
                    recent_years = year_counts.tail(2).sum()
                    total_years = year_counts.sum()
                    if recent_years / total_years > 0.6:
                        insights.append("Enforcement activity has increased in recent years")

                # Monthly patterns
                peak_month = month_counts.idxmax()
                peak_month_name = pd.to_datetime(f'2000-{peak_month:02d}-01').strftime('%B')
                insights.append(f"Peak enforcement activity in {peak_month_name}")

                # Quarterly patterns
                q4_activity = quarter_counts.get(4, 0)
                total_activity = quarter_counts.sum()
                if q4_activity / total_activity > 0.35:
                    insights.append("High Q4 activity (end-of-year enforcement push)")

                result['insights'] = insights
                temporal_results[col] = result

            except Exception as e:
                self.logger.warning(f"Could not analyze temporal patterns for {col}: {e}")
                continue

        self.analysis_results['temporal'] = temporal_results
        self.logger.info(f"Completed temporal analysis for {len(temporal_results)} variables")

        return temporal_results

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of key findings."""
        self.logger.info("Generating executive summary...")

        summary = {
            'dataset_overview': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'analysis_timestamp': datetime.now().isoformat(),
                'variable_breakdown': {
                    'categorical': len(self.categorical_vars),
                    'continuous': len(self.continuous_vars),
                    'binary': len(self.binary_vars),
                    'temporal': len(self.date_vars),
                    'text': len(self.text_vars)
                }
            },
            'key_findings': [],
            'data_quality_insights': [],
            'recommendations': []
        }

        # Data quality insights
        high_missing = [col for col, profile in self.variable_profiles.items()
                       if profile.get('null_percentage', 0) > 20]
        if high_missing:
            summary['data_quality_insights'].append(
                f"{len(high_missing)} variables have >20% missing data: {', '.join(high_missing[:3])}"
            )

        # Key findings from continuous variables
        if 'continuous' in self.analysis_results:
            for col, result in self.analysis_results['continuous'].items():
                if 'fine' in col.lower():
                    stats = result['descriptive_statistics']
                    summary['key_findings'].append(
                        f"Fine amounts: Mean €{stats['mean']:,.0f}, Median €{stats['median']:,.0f}, "
                        f"Range €{stats['min']:,.0f}-€{stats['max']:,.0f}"
                    )

                    if len(result['outliers']['extreme_outliers']) > 0:
                        summary['key_findings'].append(
                            f"Identified {len(result['outliers']['extreme_outliers'])} extreme fine outliers"
                        )

        # Key findings from categorical variables
        if 'categorical' in self.analysis_results:
            country_result = self.analysis_results['categorical'].get('A1_Country')
            if country_result:
                dominant_country = country_result['most_common_category']
                percentage = country_result['most_common_percentage']
                summary['key_findings'].append(
                    f"Enforcement dominated by {dominant_country} ({percentage:.1f}% of cases)"
                )

        # Binary variable insights
        if 'binary' in self.analysis_results:
            rare_events = [col for col, result in self.analysis_results['binary'].items()
                          if result['is_rare_event']]
            if rare_events:
                summary['key_findings'].append(
                    f"{len(rare_events)} binary variables represent rare events (<10% or >90%)"
                )

        # Recommendations
        summary['recommendations'].extend([
            "Focus analysis on continuous variables with log-transformation for skewed data",
            "Consider grouping rare categories in categorical variables for modeling",
            "Use caution with binary variables having extreme imbalance",
            "Investigate outlier cases for potential data quality issues or special circumstances"
        ])

        return summary

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete univariate analysis pipeline."""
        self.logger.info("Starting complete univariate analysis pipeline...")

        start_time = datetime.now()

        # Run all analysis components
        categorical_results = self.analyze_categorical_variables()
        continuous_results = self.analyze_continuous_variables()
        binary_results = self.analyze_binary_variables()
        temporal_results = self.analyze_temporal_patterns()

        # Generate executive summary
        executive_summary = self.generate_executive_summary()

        # Compile complete results
        complete_results = {
            'executive_summary': executive_summary,
            'variable_profiles': self.variable_profiles,
            'categorical_analysis': categorical_results,
            'continuous_analysis': continuous_results,
            'binary_analysis': binary_results,
            'temporal_analysis': temporal_results,
            'analysis_metadata': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'variables_analyzed': {
                    'categorical': len(categorical_results),
                    'continuous': len(continuous_results),
                    'binary': len(binary_results),
                    'temporal': len(temporal_results)
                }
            }
        }

        # Save results
        self.analysis_results['complete'] = complete_results

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.logger.info(f"Complete univariate analysis finished in {duration:.2f} seconds")
        self.logger.info(f"Analyzed {sum(complete_results['analysis_metadata']['variables_analyzed'].values())} variables")

        return complete_results

    def save_results(self, filepath: str):
        """Save analysis results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        self.logger.info(f"Analysis results saved to {filepath}")


def main():
    """Main execution function."""
    print("Enhanced Univariate Analysis Engine")
    print("=" * 50)

    # CLI args
    parser = argparse.ArgumentParser(description="Enhanced Univariate Analysis Engine")
    parser.add_argument('--data', '-d', dest='data_path', help='Path to cleaned CSV; defaults to latest dataNorway_cleaned_*.csv')
    args = parser.parse_args()

    # Load cleaned data
    try:
        if args.data_path:
            latest_file = args.data_path
        else:
            import glob
            cleaned_files = glob.glob('dataNorway_cleaned_*.csv')
            if not cleaned_files:
                print("❌ No cleaned data files found. Please run enhanced_data_cleaning.py first.")
                return
            latest_file = max(cleaned_files, key=lambda x: x.split('_')[-1])

        print(f"📂 Loading data from: {latest_file}")
        df = pd.read_csv(latest_file)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Initialize analyzer
    analyzer = UnivariateAnalyzer(df)

    # Run complete analysis
    print("\n🚀 Running comprehensive univariate analysis...")
    results = analyzer.run_complete_analysis()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'univariate_analysis_results_{timestamp}.json'
    analyzer.save_results(results_file)

    # Print summary
    print(f"\n✅ Analysis Complete!")
    print(f"📁 Results saved to: {results_file}")
    print(f"📊 Variables analyzed:")
    for var_type, count in results['analysis_metadata']['variables_analyzed'].items():
        print(f"  {var_type.capitalize()}: {count}")

    print(f"\n🔍 Key Findings:")
    for finding in results['executive_summary']['key_findings']:
        print(f"  • {finding}")


if __name__ == "__main__":
    main()