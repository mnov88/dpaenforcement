#!/usr/bin/env python3
"""
Automated Insight Detection and Reporting System
===============================================

Comprehensive reporting system that generates human-readable insights,
executive summaries, and detailed reports from univariate analysis results.

Author: Enhanced Data Analysis Pipeline
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

class InsightReporter:
    """Automated insight detection and report generation system."""

    def __init__(self, analysis_results: Dict[str, Any], df: pd.DataFrame):
        """Initialize reporter with analysis results and original data."""
        # Access the complete analysis results structure
        if 'complete' in analysis_results:
            self.analysis_results = analysis_results['complete']
        else:
            self.analysis_results = analysis_results

        self.df = df
        self.insights = []
        self.recommendations = []
        self.executive_insights = []

        # Report configuration
        self.significance_threshold = 0.05
        self.effect_size_threshold = 0.3
        self.outlier_threshold = 0.1  # 10% of cases as outliers is significant

        print(f"Initialized InsightReporter for {len(df)} cases")
        print(f"Using analysis results with {len(self.analysis_results)} sections")

    def detect_statistical_insights(self) -> List[Dict[str, Any]]:
        """Detect statistically significant patterns and insights."""
        print("🔍 Detecting statistical insights...")

        statistical_insights = []

        # Continuous variable insights
        if 'continuous_analysis' in self.analysis_results:
            for col, analysis in self.analysis_results['continuous_analysis'].items():
                insights = self._analyze_continuous_insights(col, analysis)
                statistical_insights.extend(insights)

        # Categorical variable insights
        if 'categorical_analysis' in self.analysis_results:
            for col, analysis in self.analysis_results['categorical_analysis'].items():
                insights = self._analyze_categorical_insights(col, analysis)
                statistical_insights.extend(insights)

        # Binary variable insights
        if 'binary_analysis' in self.analysis_results:
            binary_insights = self._analyze_binary_patterns()
            statistical_insights.extend(binary_insights)

        # Temporal insights
        if 'temporal_analysis' in self.analysis_results:
            temporal_insights = self._analyze_temporal_patterns()
            statistical_insights.extend(temporal_insights)

        print(f"  Detected {len(statistical_insights)} statistical insights")
        return statistical_insights

    def _analyze_continuous_insights(self, col: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Analyze insights for continuous variables."""
        insights = []
        stats = analysis['descriptive_statistics']

        # Distribution shape insights
        if abs(stats['skewness']) > 2:
            skew_type = "heavily right-skewed" if stats['skewness'] > 0 else "heavily left-skewed"
            insights.append({
                'type': 'distribution_shape',
                'variable': col,
                'finding': f"{col} is {skew_type}",
                'detail': f"Skewness = {stats['skewness']:.3f}",
                'significance': 'high' if abs(stats['skewness']) > 3 else 'medium',
                'implication': "Consider log transformation for modeling" if stats['skewness'] > 0 else "Consider power transformation"
            })

        # Variability insights
        cv = stats['coefficient_of_variation']
        if cv > 1.5:
            insights.append({
                'type': 'high_variability',
                'variable': col,
                'finding': f"{col} shows extremely high variability",
                'detail': f"Coefficient of variation = {cv:.2f}",
                'significance': 'high',
                'implication': "High uncertainty in outcomes; investigate underlying factors"
            })

        # Outlier insights
        outliers = analysis['outliers']
        extreme_count = len(outliers.get('extreme_outliers', []))
        total_count = stats['count']

        if extreme_count / total_count > self.outlier_threshold:
            insights.append({
                'type': 'outlier_pattern',
                'variable': col,
                'finding': f"{col} has {extreme_count} extreme outliers ({extreme_count/total_count*100:.1f}% of cases)",
                'detail': f"Outlier values: {outliers.get('extreme_outliers', [])}",
                'significance': 'high',
                'implication': "Investigate these extreme cases for data quality or special circumstances"
            })

        # Special insights for fine amounts
        if 'fine' in col.lower():
            insights.extend(self._analyze_fine_patterns(col, analysis))

        return insights

    def _analyze_fine_patterns(self, col: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Specialized analysis for fine amount patterns."""
        insights = []
        stats = analysis['descriptive_statistics']

        # Zero fine analysis
        zero_fines = (self.df[col] == 0).sum()
        total_fines = self.df[col].notna().sum()

        if zero_fines > 0:
            zero_rate = zero_fines / total_fines
            insights.append({
                'type': 'fine_pattern',
                'variable': col,
                'finding': f"{zero_fines} cases ({zero_rate*100:.1f}%) have zero fines",
                'detail': "These likely represent compliance orders without monetary penalties",
                'significance': 'medium',
                'implication': "Consider separate analysis for monetary vs non-monetary sanctions"
            })

        # Fine magnitude insights
        median_fine = stats['median']
        mean_fine = stats['mean']

        if mean_fine > 2 * median_fine:
            insights.append({
                'type': 'fine_distribution',
                'variable': col,
                'finding': f"Few very large fines dominate the distribution",
                'detail': f"Mean (€{mean_fine:,.0f}) >> Median (€{median_fine:,.0f})",
                'significance': 'high',
                'implication': "A small number of high-profile cases drive average fine amounts"
            })

        # Fine ranges
        if 'percentiles' in analysis:
            p95 = analysis['percentiles'].get('p95', 0)
            p5 = analysis['percentiles'].get('p5', 0)

            if p95 > 10 * p5 and p5 > 0:
                insights.append({
                    'type': 'fine_inequality',
                    'variable': col,
                    'finding': f"Wide range in fine amounts across cases",
                    'detail': f"95th percentile (€{p95:,.0f}) is {p95/p5:.1f}x the 5th percentile (€{p5:,.0f})",
                    'significance': 'medium',
                    'implication': "Enforcement varies significantly based on case characteristics"
                })

        return insights

    def _analyze_categorical_insights(self, col: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Analyze insights for categorical variables."""
        insights = []

        # Dominance insights
        dominance_pct = analysis['most_common_percentage']
        if dominance_pct > 80:
            insights.append({
                'type': 'category_dominance',
                'variable': col,
                'finding': f"{col} is heavily dominated by '{analysis['most_common_category']}'",
                'detail': f"{dominance_pct:.1f}% of cases",
                'significance': 'high',
                'implication': "Limited variability may reduce analytical power"
            })

        # Diversity insights
        entropy = analysis['entropy']
        unique_count = analysis['unique_categories']

        if entropy < 1 and unique_count > 2:
            insights.append({
                'type': 'low_diversity',
                'variable': col,
                'finding': f"{col} has low diversity despite {unique_count} categories",
                'detail': f"Entropy = {entropy:.3f}",
                'significance': 'medium',
                'implication': "Consider grouping rare categories for analysis"
            })

        # Rare category insights
        rare_count = analysis['rare_categories_count']
        if rare_count > 0:
            insights.append({
                'type': 'rare_categories',
                'variable': col,
                'finding': f"{col} has {rare_count} rare categories (<5% each)",
                'detail': f"Rare categories: {list(analysis['rare_categories'].keys())}",
                'significance': 'low' if rare_count < 3 else 'medium',
                'implication': "Group rare categories or use caution in statistical analysis"
            })

        return insights

    def _analyze_binary_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns across binary variables."""
        insights = []

        if 'binary_analysis' not in self.analysis_results:
            return insights

        binary_data = []
        for col, analysis in self.analysis_results['binary_analysis'].items():
            binary_data.append({
                'variable': col,
                'positive_rate': analysis['positive_rate'],
                'is_rare': analysis['is_rare_event'],
                'total_cases': analysis['total_observations']
            })

        binary_df = pd.DataFrame(binary_data)

        # Overall rare event insight
        rare_events = binary_df['is_rare'].sum()
        total_binary = len(binary_df)

        insights.append({
            'type': 'binary_overview',
            'variable': 'all_binary',
            'finding': f"{rare_events} out of {total_binary} binary variables are rare events",
            'detail': f"Rare events have <10% or >90% positive rates",
            'significance': 'medium',
            'implication': "Many binary variables may be challenging for predictive modeling"
        })

        # Identify most balanced variables
        binary_df['balance_score'] = 1 - abs(binary_df['positive_rate'] - 0.5) * 2
        balanced_vars = binary_df[binary_df['balance_score'] > 0.6].sort_values('balance_score', ascending=False)

        if len(balanced_vars) > 0:
            insights.append({
                'type': 'balanced_variables',
                'variable': 'binary_selection',
                'finding': f"Most balanced binary variables for analysis",
                'detail': f"Top balanced: {', '.join(balanced_vars['variable'].head(3).tolist())}",
                'significance': 'medium',
                'implication': "Focus on these variables for binary classification tasks"
            })

        # Legal basis patterns
        legal_basis_vars = binary_df[binary_df['variable'].str.startswith('LegalBasis_')]
        if len(legal_basis_vars) > 0:
            dominant_basis = legal_basis_vars.loc[legal_basis_vars['positive_rate'].idxmax()]
            insights.append({
                'type': 'legal_basis_pattern',
                'variable': 'legal_basis',
                'finding': f"Most common legal basis: {dominant_basis['variable'].replace('LegalBasis_', '')}",
                'detail': f"{dominant_basis['positive_rate']*100:.1f}% of cases",
                'significance': 'high',
                'implication': "Understanding dominant legal basis helps predict enforcement patterns"
            })

        return insights

    def _analyze_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Analyze temporal patterns and trends."""
        insights = []

        if 'temporal_analysis' not in self.analysis_results:
            return insights

        for col, analysis in self.analysis_results['temporal_analysis'].items():
            # Yearly trend analysis
            yearly_data = analysis['yearly_distribution']
            if len(yearly_data) > 2:
                years = sorted(yearly_data.keys())
                recent_years = years[-2:]  # Last 2 years
                earlier_years = years[:-2]

                if earlier_years:
                    recent_avg = np.mean([yearly_data[year] for year in recent_years])
                    earlier_avg = np.mean([yearly_data[year] for year in earlier_years])

                    if recent_avg > 1.5 * earlier_avg:
                        insights.append({
                            'type': 'temporal_trend',
                            'variable': col,
                            'finding': f"Significant increase in enforcement activity in recent years",
                            'detail': f"Recent average: {recent_avg:.1f} vs Earlier: {earlier_avg:.1f}",
                            'significance': 'high',
                            'implication': "GDPR enforcement is intensifying over time"
                        })

            # Seasonal patterns
            monthly_data = analysis['monthly_distribution']
            if monthly_data:
                peak_month = max(monthly_data, key=monthly_data.get)
                peak_count = monthly_data[peak_month]
                avg_monthly = np.mean(list(monthly_data.values()))

                if peak_count > 1.5 * avg_monthly:
                    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                                 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

                    insights.append({
                        'type': 'seasonal_pattern',
                        'variable': col,
                        'finding': f"Peak enforcement activity in {month_names.get(peak_month, peak_month)}",
                        'detail': f"{peak_count} cases vs {avg_monthly:.1f} average",
                        'significance': 'medium',
                        'implication': "Consider seasonal factors in enforcement planning"
                    })

        return insights

    def generate_business_insights(self) -> List[Dict[str, Any]]:
        """Generate business-context insights from the analysis."""
        print("💼 Generating business insights...")

        business_insights = []

        # GDPR enforcement effectiveness
        if 'categorical_analysis' in self.analysis_results:
            sanction_analysis = self.analysis_results['categorical_analysis'].get('A45_SanctionType')
            if sanction_analysis:
                fine_percentage = sanction_analysis['frequency_distribution'].get('Fine', 0) / sanction_analysis['total_observations'] * 100

                business_insights.append({
                    'type': 'enforcement_effectiveness',
                    'category': 'regulatory_impact',
                    'finding': f"Financial penalties used in {fine_percentage:.1f}% of enforcement cases",
                    'context': "GDPR Article 83 allows fines up to 4% of annual turnover",
                    'implication': "Monetary sanctions are a significant enforcement tool",
                    'stakeholder_impact': 'high',
                    'recommendation': "Organizations should budget for potential GDPR fines"
                })

        # Cross-border enforcement patterns
        if 'binary_analysis' in self.analysis_results:
            cross_border = self.analysis_results['binary_analysis'].get('A5_CrossBorder')
            if cross_border:
                cb_rate = cross_border['positive_percentage']

                business_insights.append({
                    'type': 'cross_border_enforcement',
                    'category': 'regulatory_complexity',
                    'finding': f"Cross-border cases represent {cb_rate:.1f}% of enforcement actions",
                    'context': "Cross-border cases involve multiple DPAs and complex coordination",
                    'implication': "International data transfers face additional scrutiny",
                    'stakeholder_impact': 'high' if cb_rate > 20 else 'medium',
                    'recommendation': "Multinational organizations need robust data governance frameworks"
                })

        # Vulnerable subject protection
        vulnerable_vars = [col for col in self.analysis_results.get('binary_analysis', {}).keys()
                          if 'VulnerableSubject_' in col]

        if vulnerable_vars:
            children_cases = self.analysis_results['binary_analysis'].get('VulnerableSubject_Children', {})
            if children_cases:
                children_rate = children_cases['positive_percentage']

                business_insights.append({
                    'type': 'vulnerable_subject_protection',
                    'category': 'compliance_priority',
                    'finding': f"Children's data involved in {children_rate:.1f}% of cases",
                    'context': "GDPR Article 8 provides enhanced protection for children's data",
                    'implication': "Special attention required for services targeting minors",
                    'stakeholder_impact': 'high',
                    'recommendation': "Implement age verification and parental consent mechanisms"
                })

        # Data subject rights enforcement
        rights_vars = [col for col in self.analysis_results.get('binary_analysis', {}).keys()
                      if 'Right_' in col]

        if rights_vars:
            access_rights = self.analysis_results['binary_analysis'].get('Right_Access', {})
            erasure_rights = self.analysis_results['binary_analysis'].get('Right_Erasure', {})

            if access_rights and erasure_rights:
                total_rights_cases = max(access_rights['positive_percentage'], erasure_rights['positive_percentage'])

                business_insights.append({
                    'type': 'data_subject_rights',
                    'category': 'operational_compliance',
                    'finding': f"Data subject rights violations appear in {total_rights_cases:.1f}% of cases",
                    'context': "GDPR Chapter III establishes comprehensive individual rights",
                    'implication': "Organizations must have robust processes for handling subject requests",
                    'stakeholder_impact': 'high',
                    'recommendation': "Implement automated systems for subject access requests"
                })

        print(f"  Generated {len(business_insights)} business insights")
        return business_insights

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        print("💡 Generating actionable recommendations...")

        recommendations = []

        # Data collection recommendations
        missing_data_vars = [col for col, profile in self.analysis_results.get('variable_profiles', {}).items()
                           if profile.get('null_percentage', 0) > 30]

        if missing_data_vars:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'recommendation': "Improve data collection for high-missing variables",
                'details': f"Focus on: {', '.join(missing_data_vars[:3])}",
                'implementation': "Review data collection processes and validation rules",
                'expected_impact': "Increased analytical power and insights accuracy"
            })

        # Statistical analysis recommendations
        if 'continuous_analysis' in self.analysis_results:
            skewed_vars = []
            for col, analysis in self.analysis_results['continuous_analysis'].items():
                if abs(analysis['descriptive_statistics']['skewness']) > 2:
                    skewed_vars.append(col)

            if skewed_vars:
                recommendations.append({
                    'type': 'statistical_modeling',
                    'priority': 'medium',
                    'recommendation': "Apply data transformations for skewed continuous variables",
                    'details': f"Variables needing transformation: {', '.join(skewed_vars)}",
                    'implementation': "Use log transformation for right-skewed data",
                    'expected_impact': "Improved model performance and assumption validity"
                })

        # Categorical analysis recommendations
        if 'categorical_analysis' in self.analysis_results:
            high_cardinality_vars = []
            for col, analysis in self.analysis_results['categorical_analysis'].items():
                if analysis['unique_categories'] > 20:
                    high_cardinality_vars.append(col)

            if high_cardinality_vars:
                recommendations.append({
                    'type': 'categorical_handling',
                    'priority': 'medium',
                    'recommendation': "Consider grouping strategies for high-cardinality categorical variables",
                    'details': f"High cardinality variables: {', '.join(high_cardinality_vars[:3])}",
                    'implementation': "Group rare categories or use embedding techniques",
                    'expected_impact': "Reduced model complexity and improved interpretability"
                })

        # Business process recommendations
        recommendations.append({
            'type': 'business_process',
            'priority': 'high',
            'recommendation': "Develop predictive models for enforcement risk assessment",
            'details': "Use balanced binary variables and transformed continuous variables",
            'implementation': "Build classification models to predict fine likelihood and amount",
            'expected_impact': "Proactive compliance management and risk mitigation"
        })

        recommendations.append({
            'type': 'regulatory_strategy',
            'priority': 'high',
            'recommendation': "Focus compliance efforts on high-impact areas",
            'details': "Prioritize data subject rights, security measures, and transparency",
            'implementation': "Allocate resources based on enforcement patterns",
            'expected_impact': "Reduced regulatory risk and improved compliance efficiency"
        })

        print(f"  Generated {len(recommendations)} actionable recommendations")
        return recommendations

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary with key insights and recommendations."""
        print("📋 Generating executive summary...")

        # Collect all insights
        statistical_insights = self.detect_statistical_insights()
        business_insights = self.generate_business_insights()
        recommendations = self.generate_recommendations()

        # Extract key metrics from the existing executive summary
        existing_summary = self.analysis_results.get('executive_summary', {})
        dataset_overview = existing_summary.get('dataset_overview', {})

        # Fine analysis summary
        fine_analysis = None
        if 'continuous_analysis' in self.analysis_results:
            for col, analysis in self.analysis_results['continuous_analysis'].items():
                if 'eur' in col.lower():
                    fine_analysis = analysis
                    break

        # Enforcement patterns
        enforcement_patterns = {}
        if 'categorical_analysis' in self.analysis_results:
            country_analysis = self.analysis_results['categorical_analysis'].get('A1_Country', {})
            sanction_analysis = self.analysis_results['categorical_analysis'].get('A45_SanctionType', {})

            if country_analysis:
                enforcement_patterns['dominant_country'] = country_analysis['most_common_category']
                enforcement_patterns['country_percentage'] = country_analysis['most_common_percentage']

            if sanction_analysis:
                enforcement_patterns['sanction_distribution'] = sanction_analysis['frequency_distribution']

        executive_summary = {
            'analysis_overview': {
                'total_cases': dataset_overview.get('total_rows', 0),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'scope': 'Norwegian GDPR Enforcement Cases',
                'time_period': '2017-2023'
            },
            'key_findings': {
                'enforcement_concentration': f"Norway dominates with {enforcement_patterns.get('country_percentage', 0):.1f}% of cases",
                'fine_patterns': self._summarize_fine_patterns(fine_analysis) if fine_analysis else "No fine data available",
                'regulatory_focus': self._identify_regulatory_focus(),
                'temporal_trends': self._summarize_temporal_trends()
            },
            'critical_insights': [insight['finding'] for insight in statistical_insights + business_insights if insight.get('significance') == 'high'][:5],
            'business_implications': [insight['implication'] for insight in business_insights][:3],
            'priority_recommendations': [rec['recommendation'] for rec in recommendations if rec['priority'] == 'high'][:3],
            'data_quality_assessment': self._assess_data_quality(),
            'next_steps': [
                "Implement predictive modeling for enforcement risk assessment",
                "Develop compliance monitoring dashboard",
                "Conduct deeper bivariate analysis of key relationships",
                "Expand dataset to include other EU countries for comparison"
            ]
        }

        return executive_summary

    def _summarize_fine_patterns(self, fine_analysis: Dict) -> str:
        """Summarize key patterns in fine amounts."""
        stats = fine_analysis['descriptive_statistics']
        outliers = len(fine_analysis['outliers'].get('extreme_outliers', []))

        return f"Mean fine €{stats['mean']:,.0f}, median €{stats['median']:,.0f}, {outliers} extreme outliers detected"

    def _identify_regulatory_focus(self) -> str:
        """Identify main areas of regulatory focus."""
        focus_areas = []

        # Check binary variables for common enforcement areas
        if 'binary_analysis' in self.analysis_results:
            security_focus = self.analysis_results['binary_analysis'].get('A29_SecurityMeasures', {})
            if security_focus and security_focus['positive_percentage'] > 30:
                focus_areas.append("Security measures")

            transparency_focus = self.analysis_results['binary_analysis'].get('A33_TransparencyObligations', {})
            if transparency_focus and transparency_focus['positive_percentage'] > 30:
                focus_areas.append("Transparency obligations")

            rights_focus = self.analysis_results['binary_analysis'].get('A31_SubjectRightsRequests', {})
            if rights_focus and rights_focus['positive_percentage'] > 20:
                focus_areas.append("Data subject rights")

        return ", ".join(focus_areas) if focus_areas else "Varied enforcement focus"

    def _summarize_temporal_trends(self) -> str:
        """Summarize temporal enforcement trends."""
        if 'temporal_analysis' not in self.analysis_results:
            return "No temporal data available"

        temporal_data = list(self.analysis_results['temporal_analysis'].values())[0]
        yearly_data = temporal_data.get('yearly_distribution', {})

        if len(yearly_data) > 2:
            years = sorted(yearly_data.keys())
            recent_trend = yearly_data[years[-1]] - yearly_data[years[-2]] if len(years) >= 2 else 0

            if recent_trend > 0:
                return f"Increasing enforcement activity (latest year: {yearly_data[years[-1]]} cases)"
            else:
                return f"Stable enforcement activity (latest year: {yearly_data[years[-1]]} cases)"

        return "Limited temporal data for trend analysis"

    def _assess_data_quality(self) -> str:
        """Assess overall data quality."""
        if 'variable_profiles' not in self.analysis_results:
            return "Data quality assessment unavailable"

        profiles = self.analysis_results['variable_profiles']
        high_quality_vars = sum(1 for profile in profiles.values() if profile.get('null_percentage', 0) < 10)
        total_vars = len(profiles)

        quality_percentage = (high_quality_vars / total_vars) * 100

        if quality_percentage > 80:
            return f"Excellent ({quality_percentage:.1f}% variables with <10% missing data)"
        elif quality_percentage > 60:
            return f"Good ({quality_percentage:.1f}% variables with <10% missing data)"
        else:
            return f"Needs improvement ({quality_percentage:.1f}% variables with <10% missing data)"

    def create_html_report(self, output_file: str) -> str:
        """Create comprehensive HTML report."""
        print("📄 Creating comprehensive HTML report...")

        # Generate all content
        executive_summary = self.generate_executive_summary()
        statistical_insights = self.detect_statistical_insights()
        business_insights = self.generate_business_insights()
        recommendations = self.generate_recommendations()

        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GDPR Enforcement Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; }
        h3 { color: #7f8c8d; margin-top: 25px; }
        .executive-summary { background-color: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .insight-card { background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .recommendation-card { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric { display: inline-block; background-color: #3498db; color: white; padding: 10px 15px; border-radius: 20px; margin: 5px; font-weight: bold; }
        .high-significance { border-left-color: #e74c3c; }
        .medium-significance { border-left-color: #f39c12; }
        .low-significance { border-left-color: #95a5a6; }
        .finding { font-weight: bold; color: #2c3e50; }
        .detail { color: #7f8c8d; font-style: italic; margin-top: 5px; }
        .implication { color: #27ae60; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .priority-high { background-color: #ffebee; }
        .priority-medium { background-color: #fff3e0; }
        .priority-low { background-color: #e8f5e8; }
        .toc { background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .toc ul { list-style-type: none; padding-left: 0; }
        .toc li { margin: 8px 0; }
        .toc a { text-decoration: none; color: #3498db; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 GDPR Enforcement Analysis Report</h1>

        <div class="executive-summary">
            <h2>📋 Executive Summary</h2>
            <div class="metric">{{ exec_summary.analysis_overview.total_cases }} Total Cases</div>
            <div class="metric">{{ exec_summary.analysis_overview.scope }}</div>
            <div class="metric">{{ exec_summary.analysis_overview.time_period }}</div>

            <h3>Key Findings</h3>
            <ul>
                {% for finding_key, finding_value in exec_summary.key_findings.items() %}
                <li><strong>{{ finding_key.replace('_', ' ').title() }}:</strong> {{ finding_value }}</li>
                {% endfor %}
            </ul>

            <h3>Data Quality</h3>
            <p><strong>Assessment:</strong> {{ exec_summary.data_quality_assessment }}</p>
        </div>

        <div class="toc">
            <h3>📑 Table of Contents</h3>
            <ul>
                <li><a href="#statistical-insights">Statistical Insights</a></li>
                <li><a href="#business-insights">Business Insights</a></li>
                <li><a href="#recommendations">Recommendations</a></li>
                <li><a href="#detailed-findings">Detailed Findings</a></li>
            </ul>
        </div>

        <section id="statistical-insights">
            <h2>📊 Statistical Insights</h2>
            {% for insight in statistical_insights %}
            <div class="insight-card {{ insight.significance }}-significance">
                <div class="finding">{{ insight.finding }}</div>
                <div class="detail">{{ insight.detail }}</div>
                <div class="implication">💡 {{ insight.implication }}</div>
                <small><strong>Variable:</strong> {{ insight.variable }} | <strong>Significance:</strong> {{ insight.significance }}</small>
            </div>
            {% endfor %}
        </section>

        <section id="business-insights">
            <h2>💼 Business Insights</h2>
            {% for insight in business_insights %}
            <div class="insight-card">
                <div class="finding">{{ insight.finding }}</div>
                <div class="detail">{{ insight.context }}</div>
                <div class="implication">💡 {{ insight.implication }}</div>
                <div style="margin-top: 10px;"><strong>Recommendation:</strong> {{ insight.recommendation }}</div>
                <small><strong>Category:</strong> {{ insight.category }} | <strong>Impact:</strong> {{ insight.stakeholder_impact }}</small>
            </div>
            {% endfor %}
        </section>

        <section id="recommendations">
            <h2>💡 Actionable Recommendations</h2>
            {% for rec in recommendations %}
            <div class="recommendation-card priority-{{ rec.priority }}">
                <div class="finding">{{ rec.recommendation }}</div>
                <div class="detail"><strong>Details:</strong> {{ rec.details }}</div>
                <div class="detail"><strong>Implementation:</strong> {{ rec.implementation }}</div>
                <div class="implication">📈 Expected Impact: {{ rec.expected_impact }}</div>
                <small><strong>Priority:</strong> {{ rec.priority.upper() }} | <strong>Type:</strong> {{ rec.type }}</small>
            </div>
            {% endfor %}
        </section>

        <section id="detailed-findings">
            <h2>📈 Detailed Analysis Results</h2>

            <h3>Variable Classification Summary</h3>
            <table>
                <tr><th>Variable Type</th><th>Count</th><th>Description</th></tr>
                <tr><td>Categorical</td><td>{{ variable_counts.categorical }}</td><td>Discrete categories (country, authority, etc.)</td></tr>
                <tr><td>Continuous</td><td>{{ variable_counts.continuous }}</td><td>Numerical values (fine amounts, etc.)</td></tr>
                <tr><td>Binary</td><td>{{ variable_counts.binary }}</td><td>Yes/No indicators</td></tr>
                <tr><td>Temporal</td><td>{{ variable_counts.temporal }}</td><td>Date/time variables</td></tr>
            </table>

            <h3>Critical Insights Summary</h3>
            <ul>
                {% for insight in exec_summary.critical_insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </ul>

            <h3>Next Steps</h3>
            <ol>
                {% for step in exec_summary.next_steps %}
                <li>{{ step }}</li>
                {% endfor %}
            </ol>
        </section>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
            <p>Generated by Enhanced GDPR Data Analysis Pipeline | {{ generation_time }}</p>
            <p>🔍 Univariate Analysis Report | 📊 Interactive Charts Available Separately</p>
        </footer>
    </div>
</body>
</html>
"""

        # Prepare template variables
        variable_counts = {
            'categorical': len(self.analysis_results.get('categorical_analysis', {})),
            'continuous': len(self.analysis_results.get('continuous_analysis', {})),
            'binary': len(self.analysis_results.get('binary_analysis', {})),
            'temporal': len(self.analysis_results.get('temporal_analysis', {}))
        }

        template_vars = {
            'exec_summary': executive_summary,
            'statistical_insights': statistical_insights,
            'business_insights': business_insights,
            'recommendations': recommendations,
            'variable_counts': variable_counts,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Render template
        template = Template(html_template)
        html_content = template.render(**template_vars)

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"  HTML report saved to: {output_file}")
        return output_file

    def save_insights_json(self, output_file: str) -> str:
        """Save all insights in structured JSON format."""
        insights_data = {
            'executive_summary': self.generate_executive_summary(),
            'statistical_insights': self.detect_statistical_insights(),
            'business_insights': self.generate_business_insights(),
            'recommendations': self.generate_recommendations(),
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_insights': len(self.detect_statistical_insights()) + len(self.generate_business_insights()),
                'analysis_scope': 'Univariate Analysis',
                'data_source': 'Norwegian GDPR Enforcement Cases'
            }
        }

        with open(output_file, 'w') as f:
            json.dump(insights_data, f, indent=2, default=str)

        print(f"📁 Insights saved to JSON: {output_file}")
        return output_file


def main():
    """Main execution function."""
    print("Automated Insight Detection and Reporting System")
    print("=" * 60)

    # Load analysis results and data
    try:
        import glob

        result_files = glob.glob('univariate_analysis_results_*.json')
        if not result_files:
            print("❌ No analysis results found. Please run univariate_analyzer.py first.")
            return

        latest_results = max(result_files, key=lambda x: x.split('_')[-1])
        print(f"📂 Loading results from: {latest_results}")

        with open(latest_results, 'r') as f:
            analysis_results = json.load(f)

        # Load data
        cleaned_files = glob.glob('dataNorway_cleaned_*.csv')
        if not cleaned_files:
            print("❌ No cleaned data files found.")
            return

        latest_data = max(cleaned_files, key=lambda x: x.split('_')[-1])
        df = pd.read_csv(latest_data)

        print(f"✓ Loaded analysis results and data")

    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Initialize reporter
    reporter = InsightReporter(analysis_results, df)

    # Generate comprehensive insights and reports
    print("\n🧠 Generating comprehensive insights and reports...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create HTML report
    html_file = f'gdpr_analysis_report_{timestamp}.html'
    reporter.create_html_report(html_file)

    # Save insights JSON
    json_file = f'analysis_insights_{timestamp}.json'
    reporter.save_insights_json(json_file)

    print(f"\n✅ Insight Detection Complete!")
    print(f"📄 HTML Report: {html_file}")
    print(f"📁 JSON Insights: {json_file}")
    print(f"🌐 Open the HTML file in your browser for the complete analysis report")


if __name__ == "__main__":
    main()