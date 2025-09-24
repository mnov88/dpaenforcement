#!/usr/bin/env python3
"""
Bivariate Insights Reporter for GDPR Enforcement Data
=====================================================

This module generates comprehensive business intelligence reports from bivariate
statistical analysis results, focusing on actionable insights for compliance
and enforcement strategy.

Features:
- Automated insight detection from statistical relationships
- Business context interpretation of statistical findings
- Risk assessment and compliance recommendations
- Interactive HTML reporting with visual context
- Executive summary generation

Author: Enhanced GDPR Data Analysis Pipeline
Version: 2.0
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

class BivariateInsightsReporter:
    """
    Comprehensive insights reporter for bivariate GDPR enforcement analysis.

    Converts statistical findings into actionable business intelligence with
    focus on regulatory compliance and enforcement risk assessment.
    """

    def __init__(self, data_path: str, bivariate_results_path: str):
        """
        Initialize the insights reporter.

        Args:
            data_path: Path to the cleaned dataset
            bivariate_results_path: Path to bivariate analysis results
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load data
        self.df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")

        # Analyze jurisdiction context
        self.jurisdiction_context = self._analyze_jurisdiction_context()
        self.logger.info(f"Jurisdiction context: {self.jurisdiction_context}")

        # Load bivariate results
        with open(bivariate_results_path, 'r') as f:
            self.bivariate_results = json.load(f)
        self.logger.info("Loaded bivariate analysis results")

        # Initialize insights
        self.insights = {
            'executive_summary': {},
            'statistical_insights': [],
            'business_insights': [],
            'risk_insights': [],
            'compliance_insights': [],
            'recommendations': [],
            'relationship_findings': {},
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_cases': len(self.df),
                'analysis_scope': f'Bivariate Analysis - {self.jurisdiction_context["scope_description"]}',
                'data_source': data_path,
                'jurisdiction_context': self.jurisdiction_context
            }
        }

    def _analyze_jurisdiction_context(self) -> Dict[str, Any]:
        """Analyze the jurisdictional context of the dataset."""
        if 'A1_Country' not in self.df.columns:
            return {
                'type': 'unknown',
                'scope_description': 'Unknown Jurisdiction Scope',
                'primary_jurisdiction': None,
                'multi_jurisdiction': False,
                'valid_for_comparison': False,
                'limitations': ['Jurisdiction information not available']
            }

        country_counts = self.df['A1_Country'].value_counts()
        total_cases = len(self.df.dropna(subset=['A1_Country']))

        # Determine primary jurisdiction
        primary_country = country_counts.index[0] if len(country_counts) > 0 else None
        primary_percentage = (country_counts.iloc[0] / total_cases * 100) if len(country_counts) > 0 else 0

        # Classify jurisdiction type
        if len(country_counts) == 1:
            jurisdiction_type = 'single_jurisdiction'
            scope_description = f'{primary_country} GDPR Enforcement Analysis'
            multi_jurisdiction = False
            valid_for_comparison = False
            limitations = [f'Single-jurisdiction study ({primary_country} only)', 'Cannot generalize to other jurisdictions']
        elif primary_percentage > 90:
            jurisdiction_type = 'quasi_single_jurisdiction'
            scope_description = f'Primarily {primary_country} GDPR Enforcement Analysis'
            multi_jurisdiction = False
            valid_for_comparison = False
            limitations = [
                f'Quasi single-jurisdiction study ({primary_percentage:.1f}% {primary_country})',
                'Insufficient cases for meaningful jurisdictional comparison',
                f'Other jurisdictions: {dict(country_counts[1:])}'
            ]
        elif primary_percentage > 70:
            jurisdiction_type = 'dominant_jurisdiction'
            scope_description = f'{primary_country}-Dominant GDPR Enforcement Analysis'
            multi_jurisdiction = True
            valid_for_comparison = False
            limitations = [
                f'Single jurisdiction dominates ({primary_percentage:.1f}% {primary_country})',
                'Limited statistical power for jurisdictional comparisons'
            ]
        else:
            jurisdiction_type = 'multi_jurisdiction'
            scope_description = 'Multi-Jurisdictional GDPR Enforcement Analysis'
            multi_jurisdiction = True
            valid_for_comparison = True
            limitations = []

        return {
            'type': jurisdiction_type,
            'scope_description': scope_description,
            'primary_jurisdiction': primary_country,
            'multi_jurisdiction': multi_jurisdiction,
            'valid_for_comparison': valid_for_comparison,
            'country_distribution': dict(country_counts),
            'primary_percentage': primary_percentage,
            'total_jurisdictions': len(country_counts),
            'limitations': limitations
        }

    def _interpret_statistical_significance(self, p_value: float, effect_size: float,
                                          sample_size: int) -> Dict[str, str]:
        """Interpret statistical significance with practical context."""
        interpretation = {
            'statistical_significance': 'not_significant',
            'practical_significance': 'negligible',
            'confidence_level': 'low',
            'sample_adequacy': 'insufficient'
        }

        # Statistical significance
        if p_value < 0.001:
            interpretation['statistical_significance'] = 'highly_significant'
            interpretation['confidence_level'] = 'very_high'
        elif p_value < 0.01:
            interpretation['statistical_significance'] = 'very_significant'
            interpretation['confidence_level'] = 'high'
        elif p_value < 0.05:
            interpretation['statistical_significance'] = 'significant'
            interpretation['confidence_level'] = 'moderate'

        # Practical significance (effect size)
        if effect_size >= 0.5:
            interpretation['practical_significance'] = 'very_large'
        elif effect_size >= 0.3:
            interpretation['practical_significance'] = 'large'
        elif effect_size >= 0.1:
            interpretation['practical_significance'] = 'medium'
        elif effect_size >= 0.02:
            interpretation['practical_significance'] = 'small'

        # Sample adequacy
        if sample_size >= 100:
            interpretation['sample_adequacy'] = 'excellent'
        elif sample_size >= 50:
            interpretation['sample_adequacy'] = 'good'
        elif sample_size >= 30:
            interpretation['sample_adequacy'] = 'adequate'
        elif sample_size >= 10:
            interpretation['sample_adequacy'] = 'limited'

        return interpretation

    def _generate_business_context(self, var1: str, var2: str, relationship_type: str) -> str:
        """Generate business context for variable relationships."""
        # Business-relevant variable mappings
        business_contexts = {
            'A46_FineAmount_EUR': 'monetary sanctions',
            'A1_Country': f'{self.jurisdiction_context["primary_jurisdiction"]} enforcement context',
            'A15_DefendantCategory': 'organization type',
            'A14_EconomicSector': 'industry sector',
            'A45_SanctionType': 'enforcement action',
            'A40_DefendantCooperation': 'compliance cooperation',
            'A38_NegligenceEstablished': 'negligence determination',
            'A16_SensitiveData': 'sensitive data processing',
            'A32_RightsInvolved': 'data subject rights',
            'A21_DataTransfers': 'international transfers',
            'A25_SubjectsAffected': 'affected individuals'
        }

        context1 = business_contexts.get(var1, var1.replace('_', ' ').lower())
        context2 = business_contexts.get(var2, var2.replace('_', ' ').lower())

        # Add jurisdiction context for single-jurisdiction studies
        jurisdiction_note = ""
        if not self.jurisdiction_context['valid_for_comparison']:
            jurisdiction_note = f" within {self.jurisdiction_context['primary_jurisdiction']} context"

        if relationship_type == 'continuous_categorical':
            return f"Relationship between {context1} and {context2} categories{jurisdiction_note}"
        elif relationship_type == 'categorical_categorical':
            return f"Association between {context1} and {context2} patterns{jurisdiction_note}"
        elif relationship_type == 'continuous_continuous':
            return f"Correlation between {context1} and {context2} magnitudes{jurisdiction_note}"
        else:
            return f"Relationship between {context1} and {context2}{jurisdiction_note}"

    def detect_statistical_insights(self) -> List[Dict[str, Any]]:
        """Detect significant statistical insights from bivariate results."""
        insights = []

        for category, relationships in self.bivariate_results.items():
            if category in ['metadata', 'summary']:
                continue

            for pair_key, result in relationships.items():
                if not isinstance(result, dict) or result.get('test_type') == 'failed':
                    continue

                # Extract variables
                if '_vs_' in pair_key:
                    var1, var2 = pair_key.split('_vs_', 1)
                else:
                    continue

                # Skip spurious A1_Country relationships in quasi-single-jurisdiction contexts
                if self._is_spurious_jurisdiction_relationship(var1, var2, result):
                    continue

                # Get statistical measures
                p_value = result.get('p_value', result.get('kw_p_value', result.get('pearson_p_value', 1)))
                effect_size = (
                    result.get('cramers_v', 0) if 'cramers_v' in result else
                    result.get('eta_squared', 0) if 'eta_squared' in result else
                    abs(result.get('pearson_r', 0)) if 'pearson_r' in result else 0
                )
                sample_size = result.get('sample_size', 0)

                # Only include significant relationships
                if p_value >= 0.05:
                    continue

                # Interpret significance
                interpretation = self._interpret_statistical_significance(p_value, effect_size, sample_size)

                # Generate business context
                business_context = self._generate_business_context(var1, var2, category)

                # Create insight
                insight = {
                    'type': 'statistical_relationship',
                    'variables': [var1, var2],
                    'relationship_type': category,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'significance_level': interpretation['statistical_significance'],
                    'practical_significance': interpretation['practical_significance'],
                    'confidence_level': interpretation['confidence_level'],
                    'business_context': business_context,
                    'statistical_details': result,
                    'finding': self._generate_finding_statement(var1, var2, result, category),
                    'implication': self._generate_implication(var1, var2, result, category, interpretation)
                }

                insights.append(insight)

        # Sort by significance and effect size
        insights.sort(key=lambda x: (x['p_value'], -x['effect_size']))

        self.logger.info(f"Detected {len(insights)} significant statistical insights (after filtering spurious relationships)")
        return insights

    def _is_spurious_jurisdiction_relationship(self, var1: str, var2: str, result: Dict) -> bool:
        """Check if a relationship with A1_Country is spurious due to quasi-single-jurisdiction context."""
        # If not involving country variable, it's not spurious
        if 'A1_Country' not in [var1, var2]:
            return False

        # If we have valid multi-jurisdiction data, relationships are not spurious
        if self.jurisdiction_context['valid_for_comparison']:
            return False

        # If quasi-single jurisdiction, A1_Country relationships are likely spurious artifacts
        if self.jurisdiction_context['type'] in ['quasi_single_jurisdiction', 'single_jurisdiction']:
            self.logger.debug(f"Filtering spurious jurisdiction relationship: {var1} vs {var2}")
            return True

        return False

    def _generate_finding_statement(self, var1: str, var2: str, result: Dict, category: str) -> str:
        """Generate clear finding statement for a relationship."""
        if category == 'categorical_categorical':
            cramers_v = result.get('cramers_v', 0)
            chi2 = result.get('chi2_statistic', 0)
            return f"Strong association between {var1} and {var2} (Cramér's V = {cramers_v:.3f}, χ² = {chi2:.2f})"

        elif category == 'continuous_categorical':
            eta_squared = result.get('eta_squared', 0)
            kw_stat = result.get('kw_statistic', 0)
            return f"Significant group differences in {var1} across {var2} categories (η² = {eta_squared:.3f}, H = {kw_stat:.2f})"

        elif category == 'continuous_continuous':
            pearson_r = result.get('pearson_r', 0)
            spearman_r = result.get('spearman_r', 0)
            primary_r = spearman_r if abs(pearson_r - spearman_r) > 0.1 else pearson_r
            return f"Significant correlation between {var1} and {var2} (r = {primary_r:.3f})"

        return f"Significant relationship between {var1} and {var2}"

    def _generate_implication(self, var1: str, var2: str, result: Dict, category: str,
                            interpretation: Dict) -> str:
        """Generate business implication for a relationship."""
        # Fine amount implications
        if 'FineAmount' in var1 or 'FineAmount' in var2:
            if 'DefendantCooperation' in var1 or 'DefendantCooperation' in var2:
                return "Cooperation level significantly impacts fine amounts - incentivizes collaborative compliance"
            elif 'EconomicSector' in var1 or 'EconomicSector' in var2:
                return "Sector-specific enforcement patterns suggest targeted compliance strategies needed"
            elif 'SensitiveData' in var1 or 'SensitiveData' in var2:
                return "Sensitive data processing cases face higher financial penalties"

        # Sanction type implications
        if 'SanctionType' in var1 or 'SanctionType' in var2:
            if 'Country' in var1 or 'Country' in var2:
                if self.jurisdiction_context['valid_for_comparison']:
                    return "Enforcement approach varies by jurisdiction - requires country-specific compliance strategies"
                else:
                    return f"Sanction patterns within {self.jurisdiction_context['primary_jurisdiction']} enforcement framework"
            elif 'DefendantCategory' in var1 or 'DefendantCategory' in var2:
                return "Organization type influences sanction selection - tailor compliance by entity type"

        # Data subject rights implications
        if 'RightsInvolved' in var1 or 'RightsInvolved' in var2:
            return "Data subject rights violations strongly correlate with enforcement actions"

        # General implications based on effect size
        if interpretation['practical_significance'] in ['large', 'very_large']:
            return "Strong relationship suggests this factor is a key enforcement predictor"
        elif interpretation['practical_significance'] == 'medium':
            return "Moderate relationship indicates important compliance consideration"
        else:
            return "Relationship detected but practical impact may be limited"

    def generate_business_insights(self) -> List[Dict[str, Any]]:
        """Generate business-focused insights from statistical findings."""
        business_insights = []

        # Fine determinant analysis
        fine_relationships = []
        for category, relationships in self.bivariate_results.items():
            for pair_key, result in relationships.items():
                if ('FineAmount' in pair_key and isinstance(result, dict) and
                    result.get('p_value', result.get('kw_p_value', 1)) < 0.05):
                    fine_relationships.append((pair_key, result))

        if fine_relationships:
            business_insights.append({
                'type': 'fine_determinants',
                'title': 'Key Fine Amount Predictors Identified',
                'finding': f"Analysis reveals {len(fine_relationships)} significant factors influencing fine amounts",
                'details': [self._extract_fine_insight(pair, result) for pair, result in fine_relationships[:5]],
                'business_impact': 'High',
                'recommendation': 'Focus compliance efforts on identified high-risk factors'
            })

        # Sector-specific patterns
        sector_relationships = []
        for category, relationships in self.bivariate_results.items():
            for pair_key, result in relationships.items():
                if ('EconomicSector' in pair_key and isinstance(result, dict) and
                    result.get('p_value', result.get('chi2_statistic', 1)) < 0.05):
                    sector_relationships.append((pair_key, result))

        if sector_relationships:
            business_insights.append({
                'type': 'sector_patterns',
                'title': 'Industry-Specific Enforcement Patterns',
                'finding': f"Significant enforcement variations across economic sectors",
                'details': "Certain industries face disproportionate scrutiny and penalties",
                'business_impact': 'High',
                'recommendation': 'Develop sector-specific compliance strategies'
            })

        # Cooperation impact analysis
        cooperation_relationships = []
        for category, relationships in self.bivariate_results.items():
            for pair_key, result in relationships.items():
                if ('DefendantCooperation' in pair_key and isinstance(result, dict) and
                    result.get('p_value', result.get('kw_p_value', 1)) < 0.05):
                    cooperation_relationships.append((pair_key, result))

        if cooperation_relationships:
            business_insights.append({
                'type': 'cooperation_impact',
                'title': 'Cooperation Significantly Affects Outcomes',
                'finding': 'Defendant cooperation level strongly correlates with enforcement outcomes',
                'details': 'Cooperative defendants receive more favorable treatment',
                'business_impact': 'Medium',
                'recommendation': 'Implement proactive cooperation protocols in compliance program'
            })

        # Jurisdiction analysis - only valid for multi-jurisdiction datasets
        if self.jurisdiction_context['valid_for_comparison']:
            country_relationships = []
            for category, relationships in self.bivariate_results.items():
                for pair_key, result in relationships.items():
                    if ('Country' in pair_key and isinstance(result, dict) and
                        result.get('p_value', result.get('chi2_statistic', 1)) < 0.05):
                        country_relationships.append((pair_key, result))

            if country_relationships:
                business_insights.append({
                    'type': 'jurisdictional_variance',
                    'title': 'Significant Jurisdictional Enforcement Differences',
                    'finding': 'Enforcement patterns vary substantially across countries',
                    'details': 'Different jurisdictions show distinct enforcement philosophies and penalty structures',
                    'business_impact': 'High',
                    'recommendation': 'Develop country-specific compliance strategies for multinational operations'
                })
        else:
            # Add limitation insight for single/quasi-single jurisdiction studies
            business_insights.append({
                'type': 'jurisdiction_limitation',
                'title': f'Analysis Limited to {self.jurisdiction_context["primary_jurisdiction"]} Context',
                'finding': f'Study focuses on {self.jurisdiction_context["primary_jurisdiction"]} enforcement patterns',
                'details': f'{"; ".join(self.jurisdiction_context["limitations"])}',
                'business_impact': 'Medium',
                'recommendation': f'Apply insights specifically to {self.jurisdiction_context["primary_jurisdiction"]} compliance strategy; exercise caution when generalizing to other jurisdictions'
            })

        self.logger.info(f"Generated {len(business_insights)} business insights")
        return business_insights

    def _extract_fine_insight(self, pair_key: str, result: Dict) -> str:
        """Extract specific insight about fine amount relationships."""
        var1, var2 = pair_key.split('_vs_', 1)
        other_var = var2 if 'FineAmount' in var1 else var1

        if 'group_statistics' in result:
            groups = result['group_statistics']
            if len(groups) >= 2:
                group_means = {k: v['mean'] for k, v in groups.items() if 'mean' in v}
                if group_means:
                    highest = max(group_means, key=group_means.get)
                    lowest = min(group_means, key=group_means.get)
                    return f"{other_var}: {highest} category shows highest fines (€{group_means[highest]:,.0f} avg)"

        effect_size = result.get('eta_squared', result.get('cramers_v', 0))
        return f"{other_var}: Explains {effect_size*100:.1f}% of fine amount variation"

    def generate_risk_insights(self) -> List[Dict[str, Any]]:
        """Generate risk-focused insights for compliance planning."""
        risk_insights = []

        # High-risk factor identification
        high_risk_factors = []
        for category, relationships in self.bivariate_results.items():
            for pair_key, result in relationships.items():
                if not isinstance(result, dict):
                    continue

                p_value = result.get('p_value', result.get('kw_p_value', result.get('pearson_p_value', 1)))
                effect_size = (
                    result.get('cramers_v', 0) if 'cramers_v' in result else
                    result.get('eta_squared', 0) if 'eta_squared' in result else
                    abs(result.get('pearson_r', 0)) if 'pearson_r' in result else 0
                )

                # High-risk criteria: significant and medium+ effect
                if p_value < 0.05 and effect_size > 0.1:
                    if any(risk_term in pair_key for risk_term in
                          ['FineAmount', 'SanctionType', 'NegligenceEstablished', 'SensitiveData']):
                        high_risk_factors.append({
                            'relationship': pair_key,
                            'risk_score': (1 - p_value) * effect_size,
                            'effect_size': effect_size,
                            'p_value': p_value
                        })

        # Sort by risk score
        high_risk_factors.sort(key=lambda x: x['risk_score'], reverse=True)

        if high_risk_factors:
            risk_insights.append({
                'type': 'high_risk_factors',
                'title': 'Critical Risk Factors Identified',
                'finding': f"Analysis identifies {len(high_risk_factors)} high-risk relationships",
                'top_risks': [factor['relationship'] for factor in high_risk_factors[:5]],
                'risk_level': 'High',
                'mitigation': 'Prioritize monitoring and controls for identified risk factors'
            })

        # Sensitive data risk profile
        sensitive_data_risks = []
        for category, relationships in self.bivariate_results.items():
            for pair_key, result in relationships.items():
                if ('SensitiveData' in pair_key and isinstance(result, dict) and
                    result.get('p_value', 1) < 0.05):
                    sensitive_data_risks.append(pair_key)

        if sensitive_data_risks:
            risk_insights.append({
                'type': 'sensitive_data_risk',
                'title': 'Sensitive Data Processing Risk Profile',
                'finding': 'Sensitive data processing significantly correlates with enforcement actions',
                'affected_relationships': len(sensitive_data_risks),
                'risk_level': 'Very High',
                'mitigation': 'Implement enhanced controls for all sensitive data processing activities'
            })

        self.logger.info(f"Generated {len(risk_insights)} risk insights")
        return risk_insights

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on bivariate findings."""
        recommendations = []

        # Get summary statistics
        summary = self.bivariate_results.get('summary', {})
        significant_count = summary.get('significant_relationships', 0)
        strong_count = summary.get('strong_relationships', 0)

        # Strategic recommendations
        if significant_count > 0:
            recommendations.append({
                'type': 'strategic',
                'priority': 'high',
                'recommendation': 'Implement relationship-based risk assessment model',
                'details': f'Analysis reveals {significant_count} significant relationships that can predict enforcement outcomes',
                'implementation': 'Develop scoring system using identified predictive relationships',
                'expected_impact': 'Proactive identification of high-risk scenarios and improved compliance allocation'
            })

        # Fine mitigation recommendations
        fine_relationships = sum(1 for cat, rels in self.bivariate_results.items()
                               if cat != 'summary' and cat != 'metadata'
                               for pair, result in rels.items()
                               if isinstance(result, dict) and 'FineAmount' in pair and
                               result.get('p_value', result.get('kw_p_value', 1)) < 0.05)

        if fine_relationships > 0:
            recommendations.append({
                'type': 'operational',
                'priority': 'high',
                'recommendation': 'Optimize fine mitigation strategies',
                'details': f'Identified {fine_relationships} factors significantly influencing fine amounts',
                'implementation': 'Focus compliance efforts on cooperation, sector-specific risks, and data sensitivity levels',
                'expected_impact': 'Reduced financial exposure and improved regulatory relationships'
            })

        # Sector-specific recommendations
        sector_relationships = sum(1 for cat, rels in self.bivariate_results.items()
                                 if cat != 'summary' and cat != 'metadata'
                                 for pair, result in rels.items()
                                 if isinstance(result, dict) and 'EconomicSector' in pair and
                                 result.get('p_value', result.get('chi2_statistic', 1)) < 0.05)

        if sector_relationships > 0:
            recommendations.append({
                'type': 'tactical',
                'priority': 'medium',
                'recommendation': 'Develop industry-specific compliance frameworks',
                'details': 'Different economic sectors show distinct enforcement patterns and risks',
                'implementation': 'Create sector-tailored compliance programs and risk assessments',
                'expected_impact': 'More effective resource allocation and targeted risk mitigation'
            })

        # Data governance recommendations
        if any('SensitiveData' in pair for cat, rels in self.bivariate_results.items()
               if cat not in ['summary', 'metadata']
               for pair in rels.keys()):
            recommendations.append({
                'type': 'governance',
                'priority': 'high',
                'recommendation': 'Enhance sensitive data governance framework',
                'details': 'Sensitive data processing shows significant correlation with enforcement actions',
                'implementation': 'Implement enhanced controls, monitoring, and approval processes for sensitive data',
                'expected_impact': 'Reduced enforcement risk and improved data subject protection'
            })

        # Cooperation protocol recommendations
        cooperation_relationships = sum(1 for cat, rels in self.bivariate_results.items()
                                      if cat != 'summary' and cat != 'metadata'
                                      for pair, result in rels.items()
                                      if isinstance(result, dict) and 'Cooperation' in pair and
                                      result.get('p_value', result.get('kw_p_value', 1)) < 0.05)

        if cooperation_relationships > 0:
            recommendations.append({
                'type': 'procedural',
                'priority': 'medium',
                'recommendation': 'Establish proactive regulatory cooperation protocols',
                'details': 'Cooperation level significantly affects enforcement outcomes',
                'implementation': 'Develop incident response procedures emphasizing transparency and collaboration',
                'expected_impact': 'Improved regulatory relationships and more favorable enforcement outcomes'
            })

        self.logger.info(f"Generated {len(recommendations)} actionable recommendations")
        return recommendations

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive executive summary."""
        summary_stats = self.bivariate_results.get('summary', {})

        # Key findings
        total_relationships = summary_stats.get('total_relationships_analyzed', 0)
        significant_relationships = summary_stats.get('significant_relationships', 0)
        strong_relationships = summary_stats.get('strong_relationships', 0)

        # Calculate percentages
        sig_percentage = (significant_relationships / total_relationships * 100) if total_relationships > 0 else 0
        strong_percentage = (strong_relationships / total_relationships * 100) if total_relationships > 0 else 0

        # Generate key insights
        key_insights = []
        if sig_percentage > 20:
            key_insights.append(f"High interconnectedness: {sig_percentage:.1f}% of relationships show statistical significance")
        if strong_relationships > 0:
            key_insights.append(f"Strong predictive patterns: {strong_relationships} relationships show large effect sizes")

        # Business implications
        business_implications = []
        if any('FineAmount' in pair for cat, rels in self.bivariate_results.items()
               if cat not in ['summary', 'metadata'] for pair in rels.keys()):
            business_implications.append("Fine amount predictors identified - enables proactive risk management")
        if any('Cooperation' in pair for cat, rels in self.bivariate_results.items()
               if cat not in ['summary', 'metadata'] for pair in rels.keys()):
            business_implications.append("Cooperation impact quantified - incentivizes collaborative compliance")

        summary = {
            'analysis_overview': {
                'total_relationships_analyzed': total_relationships,
                'significant_relationships': significant_relationships,
                'strong_relationships': strong_relationships,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'scope': 'Bivariate Relationship Analysis',
                'time_period': '2017-2023'
            },
            'key_findings': {
                'statistical_significance': f"{sig_percentage:.1f}% of relationships statistically significant",
                'practical_significance': f"{strong_relationships} relationships with large effect sizes",
                'predictive_value': "Multiple factors identified as enforcement outcome predictors",
                'business_relevance': "Clear patterns for compliance strategy optimization"
            },
            'critical_insights': key_insights,
            'business_implications': business_implications,
            'priority_recommendations': [
                "Implement relationship-based risk scoring system",
                "Focus compliance on cooperation and sensitive data controls",
                "Develop sector-specific enforcement response strategies"
            ],
            'confidence_assessment': f"High (based on {len(self.df)} cases with robust statistical analysis)",
            'next_steps': self._generate_next_steps()
        }

        return summary

    def _generate_next_steps(self) -> List[str]:
        """Generate appropriate next steps based on jurisdiction context."""
        base_steps = [
            "Implement predictive modeling using identified relationships",
            "Develop automated risk scoring dashboard",
            "Conduct temporal analysis of relationship stability"
        ]

        if self.jurisdiction_context['valid_for_comparison']:
            base_steps.append("Expand analysis to include additional EU jurisdictions for comparison")
        else:
            base_steps.extend([
                f"Validate findings with additional {self.jurisdiction_context['primary_jurisdiction']} case data",
                "Exercise caution when applying insights to other jurisdictions",
                "Consider multi-jurisdictional study design for broader applicability"
            ])

        return base_steps

    def create_html_report(self) -> str:
        """Create comprehensive HTML report."""
        # Generate all insights
        statistical_insights = self.detect_statistical_insights()
        business_insights = self.generate_business_insights()
        risk_insights = self.generate_risk_insights()
        recommendations = self.generate_recommendations()
        executive_summary = self.generate_executive_summary()

        # Store insights
        self.insights['statistical_insights'] = statistical_insights
        self.insights['business_insights'] = business_insights
        self.insights['risk_insights'] = risk_insights
        self.insights['recommendations'] = recommendations
        self.insights['executive_summary'] = executive_summary

        # Create HTML content
        html_content = self._generate_html_template()

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"bivariate_analysis_report_{timestamp}.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"HTML report saved to: {output_path}")
        return output_path

    def _generate_html_template(self) -> str:
        """Generate HTML template with insights."""
        summary = self.insights['executive_summary']

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.jurisdiction_context['scope_description']} - Bivariate Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; margin-top: 25px; }}
        .executive-summary {{ background-color: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .insight-card {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .risk-card {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .business-card {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendation-card {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 15px; border-radius: 20px; margin: 5px; font-weight: bold; }}
        .high-priority {{ background-color: #e74c3c; }}
        .medium-priority {{ background-color: #f39c12; }}
        .low-priority {{ background-color: #95a5a6; }}
        .finding {{ font-weight: bold; color: #2c3e50; }}
        .detail {{ color: #7f8c8d; margin-top: 5px; }}
        .implication {{ color: #27ae60; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .toc {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc li {{ margin: 8px 0; }}
        .toc a {{ text-decoration: none; color: #3498db; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔗 {self.jurisdiction_context['scope_description']} - Bivariate Relationship Analysis</h1>

        <div class="executive-summary">
            <h2>📋 Executive Summary</h2>
            <div class="metric">{summary['analysis_overview']['total_relationships_analyzed']} Relationships Analyzed</div>
            <div class="metric">{summary['analysis_overview']['significant_relationships']} Significant</div>
            <div class="metric">{summary['analysis_overview']['strong_relationships']} Strong Effects</div>
            <div class="metric">{summary['analysis_overview']['scope']}</div>

            <h3>Key Findings</h3>
            <ul>
                <li><strong>Statistical Significance:</strong> {summary['key_findings']['statistical_significance']}</li>
                <li><strong>Practical Significance:</strong> {summary['key_findings']['practical_significance']}</li>
                <li><strong>Predictive Value:</strong> {summary['key_findings']['predictive_value']}</li>
                <li><strong>Business Relevance:</strong> {summary['key_findings']['business_relevance']}</li>
            </ul>

            <h3>Confidence Assessment</h3>
            <p><strong>Assessment:</strong> {summary['confidence_assessment']}</p>

            <h3>Study Limitations</h3>
            <div class="limitation-box" style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 10px 0;">
                <strong>Jurisdictional Scope:</strong> {self.jurisdiction_context['scope_description']}<br>
                {'<br>'.join([f'• {limitation}' for limitation in self.jurisdiction_context['limitations']])}
            </div>
        </div>

        <div class="toc">
            <h3>📑 Table of Contents</h3>
            <ul>
                <li><a href="#statistical-insights">Statistical Insights</a></li>
                <li><a href="#business-insights">Business Intelligence</a></li>
                <li><a href="#risk-insights">Risk Assessment</a></li>
                <li><a href="#recommendations">Recommendations</a></li>
                <li><a href="#detailed-findings">Detailed Findings</a></li>
            </ul>
        </div>

        <section id="statistical-insights">
            <h2>📊 Statistical Insights</h2>
"""

        # Add statistical insights
        for insight in self.insights['statistical_insights'][:10]:  # Top 10
            significance_class = "high-priority" if insight['p_value'] < 0.01 else "medium-priority" if insight['p_value'] < 0.05 else "low-priority"
            html += f"""
            <div class="insight-card {significance_class}">
                <div class="finding">{insight['finding']}</div>
                <div class="detail">{insight['business_context']}</div>
                <div class="implication">💡 {insight['implication']}</div>
                <small><strong>p-value:</strong> {insight['p_value']:.4f} | <strong>Effect Size:</strong> {insight['effect_size']:.3f} | <strong>Sample:</strong> {insight['sample_size']}</small>
            </div>
"""

        html += """
        </section>

        <section id="business-insights">
            <h2>💼 Business Intelligence</h2>
"""

        # Add business insights
        for insight in self.insights['business_insights']:
            html += f"""
            <div class="business-card">
                <div class="finding">{insight['title']}</div>
                <div class="detail"><strong>Finding:</strong> {insight['finding']}</div>
                <div class="detail"><strong>Impact:</strong> {insight['business_impact']}</div>
                <div class="implication">📈 Recommendation: {insight['recommendation']}</div>
            </div>
"""

        html += """
        </section>

        <section id="risk-insights">
            <h2>⚠️ Risk Assessment</h2>
"""

        # Add risk insights
        for insight in self.insights['risk_insights']:
            html += f"""
            <div class="risk-card">
                <div class="finding">{insight['title']}</div>
                <div class="detail"><strong>Finding:</strong> {insight['finding']}</div>
                <div class="detail"><strong>Risk Level:</strong> {insight['risk_level']}</div>
                <div class="implication">🛡️ Mitigation: {insight.get('mitigation', 'Monitor and assess')}</div>
            </div>
"""

        html += """
        </section>

        <section id="recommendations">
            <h2>💡 Actionable Recommendations</h2>
"""

        # Add recommendations
        for rec in self.insights['recommendations']:
            priority_class = f"{rec['priority']}-priority"
            html += f"""
            <div class="recommendation-card {priority_class}">
                <div class="finding">{rec['recommendation']}</div>
                <div class="detail"><strong>Details:</strong> {rec['details']}</div>
                <div class="detail"><strong>Implementation:</strong> {rec['implementation']}</div>
                <div class="implication">📈 Expected Impact: {rec['expected_impact']}</div>
                <small><strong>Priority:</strong> {rec['priority'].upper()} | <strong>Type:</strong> {rec['type']}</small>
            </div>
"""

        html += f"""
        </section>

        <section id="detailed-findings">
            <h2>📈 Detailed Analysis Results</h2>

            <h3>Relationship Analysis Summary</h3>
            <table>
                <tr><th>Analysis Type</th><th>Count</th><th>Description</th></tr>
                <tr><td>Categorical-Categorical</td><td>{summary['analysis_overview'].get('categorical_categorical', 0)}</td><td>Chi-square tests with Cramér's V effect sizes</td></tr>
                <tr><td>Continuous-Categorical</td><td>{summary['analysis_overview'].get('continuous_categorical', 0)}</td><td>Kruskal-Wallis tests with eta-squared effect sizes</td></tr>
                <tr><td>Continuous-Continuous</td><td>{summary['analysis_overview'].get('continuous_continuous', 0)}</td><td>Correlation analysis with significance testing</td></tr>
            </table>

            <h3>Critical Insights Summary</h3>
            <ul>
"""

        for insight in summary.get('critical_insights', []):
            html += f"<li>{insight}</li>"

        html += f"""
            </ul>

            <h3>Next Steps</h3>
            <ol>
"""

        for step in summary.get('next_steps', []):
            html += f"<li>{step}</li>"

        html += f"""
            </ol>
        </section>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
            <p>Generated by Enhanced GDPR Data Analysis Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🔗 {self.jurisdiction_context['scope_description']} | 📊 Statistical Relationships and Business Intelligence</p>
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; margin: 10px 0; font-size: 0.9em;">
                <strong>⚠️ Methodological Note:</strong> This analysis is based on {len(self.df)} cases from {self.jurisdiction_context['primary_jurisdiction']}
                ({self.jurisdiction_context['primary_percentage']:.1f}% of dataset).
                Findings are specific to {self.jurisdiction_context['primary_jurisdiction']} enforcement context and should not be generalized to other jurisdictions without additional validation.
            </div>
        </footer>
    </div>
</body>
</html>
"""

        return html

    def save_insights_json(self) -> str:
        """Save insights to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"bivariate_insights_{timestamp}.json"

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        converted_insights = convert_types(self.insights)

        with open(output_path, 'w') as f:
            json.dump(converted_insights, f, indent=2)

        self.logger.info(f"Insights saved to JSON: {output_path}")
        return output_path

def main():
    """Main execution function."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Bivariate Insights Reporter")
    print("=" * 35)

    # Find the most recent data and bivariate results files
    data_files = list(Path('.').glob('dataNorway_cleaned_*.csv'))
    bivariate_files = list(Path('.').glob('bivariate_analysis_results_*.json'))

    if not data_files:
        print("❌ No cleaned data files found.")
        sys.exit(1)

    if not bivariate_files:
        print("❌ No bivariate analysis results found. Please run bivariate_analyzer.py first.")
        sys.exit(1)

    data_path = str(sorted(data_files)[-1])
    bivariate_path = str(sorted(bivariate_files)[-1])

    print(f"📂 Loading data from: {data_path}")
    print(f"📊 Loading bivariate results from: {bivariate_path}")

    # Initialize insights reporter
    reporter = BivariateInsightsReporter(data_path, bivariate_path)

    # Generate comprehensive insights report
    print("🧠 Generating comprehensive insights and reports...")
    html_path = reporter.create_html_report()
    json_path = reporter.save_insights_json()

    # Summary statistics
    statistical_insights = len(reporter.insights['statistical_insights'])
    business_insights = len(reporter.insights['business_insights'])
    risk_insights = len(reporter.insights['risk_insights'])
    recommendations = len(reporter.insights['recommendations'])

    print("\n✅ Bivariate Insights Generation Complete!")
    print(f"📄 HTML Report: {html_path}")
    print(f"📁 JSON Insights: {json_path}")
    print(f"📊 Insights generated:")
    print(f"  Statistical: {statistical_insights}")
    print(f"  Business: {business_insights}")
    print(f"  Risk: {risk_insights}")
    print(f"  Recommendations: {recommendations}")
    print("🌐 Open the HTML file in your browser for the complete analysis report")

if __name__ == "__main__":
    main()