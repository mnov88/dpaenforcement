#!/usr/bin/env python3
"""
Bivariate Visualization Engine for GDPR Enforcement Data
========================================================

This module creates comprehensive interactive visualizations for bivariate
relationships in GDPR enforcement data, supporting all relationship types
with appropriate statistical context.

Features:
- Correlation heatmaps with significance indicators
- Interactive scatter plots with grouping and regression lines
- Cross-tabulation visualizations with statistical annotations
- Box plots and violin plots for group comparisons
- Network-style relationship diagrams
- Statistical significance overlays

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
from typing import Dict, List, Tuple, Optional, Any

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.figure_factory as ff

# Statistical libraries for visualization support
from scipy import stats
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BivariateVisualizationEngine:
    """
    Comprehensive visualization engine for bivariate relationships in GDPR data.

    Creates interactive visualizations that showcase statistical relationships
    between variables with appropriate context and business intelligence focus.
    """

    def __init__(self, data_path: str, bivariate_results_path: str):
        """
        Initialize the visualization engine.

        Args:
            data_path: Path to the cleaned dataset
            bivariate_results_path: Path to bivariate analysis results
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load data
        self.df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")

        # Load bivariate results
        with open(bivariate_results_path, 'r') as f:
            self.bivariate_results = json.load(f)
        self.logger.info("Loaded bivariate analysis results")

        # Create output directory
        self.output_dir = Path("bivariate_visualizations")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize chart tracking
        self.charts_created = []

    def create_correlation_heatmap(self) -> str:
        """Create interactive correlation heatmap for continuous variables."""
        continuous_vars = ['A46_FineAmount', 'A46_FineAmount_EUR']
        available_vars = [var for var in continuous_vars if var in self.df.columns]

        if len(available_vars) < 2:
            self.logger.warning("Insufficient continuous variables for correlation heatmap")
            return None

        # Calculate correlation matrix
        corr_data = self.df[available_vars].corr()

        # Create correlation p-values matrix
        p_values = np.ones_like(corr_data)
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i != j and var1 in self.df.columns and var2 in self.df.columns:
                    data_clean = self.df[[var1, var2]].dropna()
                    if len(data_clean) > 2:
                        _, p_val = stats.pearsonr(data_clean[var1], data_clean[var2])
                        p_values[i, j] = p_val

        # Create annotations with correlation values and significance
        annotations = []
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                correlation = corr_data.iloc[i, j]
                p_val = p_values[i, j]
                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{correlation:.3f}{sig_marker}",
                        showarrow=False,
                        font=dict(color="white" if abs(correlation) > 0.5 else "black", size=12)
                    )
                )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=available_vars,
            y=available_vars,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Correlation Coefficient"),
            hoverongaps=False
        ))

        fig.update_layout(
            title="Correlation Matrix for Continuous Variables<br><sub>*p<0.05, **p<0.01, ***p<0.001</sub>",
            xaxis_title="Variables",
            yaxis_title="Variables",
            width=600,
            height=500
        )

        # Add annotations
        fig.update_layout(annotations=annotations)

        # Save chart
        output_path = self.output_dir / "correlation_heatmap.html"
        fig.write_html(str(output_path))
        self.charts_created.append("correlation_heatmap")
        self.logger.info("Created correlation heatmap")
        return str(output_path)

    def create_fine_amount_analysis(self) -> List[str]:
        """Create comprehensive fine amount relationship visualizations."""
        charts = []

        if 'A46_FineAmount_EUR' not in self.df.columns:
            self.logger.warning("Fine amount data not available")
            return charts

        # Key categorical variables for fine analysis
        key_categoricals = [
            'A1_Country', 'A15_DefendantCategory', 'A14_EconomicSector',
            'A45_SanctionType', 'A40_DefendantCooperation', 'A38_NegligenceEstablished'
        ]

        for cat_var in key_categoricals:
            if cat_var not in self.df.columns:
                continue

            # Create box plot with statistical annotations
            fig = self._create_enhanced_boxplot('A46_FineAmount_EUR', cat_var)
            if fig:
                output_path = self.output_dir / f"fine_analysis_{cat_var}.html"
                fig.write_html(str(output_path))
                charts.append(f"fine_analysis_{cat_var}")
                self.logger.info(f"Created fine analysis chart for {cat_var}")

        return charts

    def _create_enhanced_boxplot(self, continuous_var: str, categorical_var: str) -> Optional[go.Figure]:
        """Create enhanced box plot with statistical context."""
        data = self.df[[continuous_var, categorical_var]].dropna()

        if len(data) < 10:
            return None

        # Transform fine amounts for better visualization (log scale)
        data_viz = data.copy()
        data_viz[f'{continuous_var}_log'] = np.log1p(data_viz[continuous_var])

        # Create box plot
        fig = px.box(
            data,
            x=categorical_var,
            y=continuous_var,
            title=f"Fine Amounts by {categorical_var}",
            labels={continuous_var: "Fine Amount (EUR)", categorical_var: categorical_var.replace('_', ' ')},
            hover_data=[continuous_var]
        )

        # Add statistical annotations if bivariate results available
        result_key = f"{continuous_var}_vs_{categorical_var}"
        if result_key in self.bivariate_results.get('continuous_categorical', {}):
            result = self.bivariate_results['continuous_categorical'][result_key]
            p_value = result.get('kw_p_value', 1)
            eta_squared = result.get('eta_squared', 0)

            # Add statistical annotation
            fig.add_annotation(
                text=f"Kruskal-Wallis p = {p_value:.4f}<br>η² = {eta_squared:.3f}<br>{result.get('interpretation', '')}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )

        # Update layout
        fig.update_layout(
            width=800,
            height=500,
            xaxis_tickangle=-45
        )

        return fig

    def create_categorical_relationships_grid(self) -> str:
        """Create grid of key categorical relationship visualizations."""
        # Select most interesting categorical relationships
        key_relationships = [
            ('A1_Country', 'A45_SanctionType'),
            ('A15_DefendantCategory', 'A45_SanctionType'),
            ('A40_DefendantCooperation', 'A38_NegligenceEstablished'),
            ('A16_SensitiveData', 'A45_SanctionType')
        ]

        # Filter to available variables
        available_relationships = [
            (var1, var2) for var1, var2 in key_relationships
            if var1 in self.df.columns and var2 in self.df.columns
        ]

        if not available_relationships:
            self.logger.warning("No key categorical relationships available")
            return None

        # Create subplot grid
        rows = 2
        cols = 2
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{v1} vs {v2}" for v1, v2 in available_relationships[:4]],
            specs=[[{"type": "bar"} for _ in range(cols)] for _ in range(rows)]
        )

        for idx, (var1, var2) in enumerate(available_relationships[:4]):
            row = idx // cols + 1
            col = idx % cols + 1

            # Create cross-tabulation
            crosstab = pd.crosstab(self.df[var1], self.df[var2], normalize='index') * 100

            # Add stacked bar chart
            for i, category in enumerate(crosstab.columns):
                fig.add_trace(
                    go.Bar(
                        name=str(category),
                        x=crosstab.index,
                        y=crosstab[category],
                        showlegend=(idx == 0),  # Only show legend for first subplot
                        hovertemplate=f"{category}: %{{y:.1f}}%<extra></extra>"
                    ),
                    row=row, col=col
                )

        # Update layout
        fig.update_layout(
            title="Key Categorical Relationships in GDPR Enforcement",
            barmode='stack',
            height=800,
            width=1200,
            showlegend=True
        )

        # Save chart
        output_path = self.output_dir / "categorical_relationships_grid.html"
        fig.write_html(str(output_path))
        self.charts_created.append("categorical_relationships_grid")
        self.logger.info("Created categorical relationships grid")
        return str(output_path)

    def create_enforcement_pattern_analysis(self) -> str:
        """Create comprehensive enforcement pattern visualization."""
        if 'A46_FineAmount_EUR' not in self.df.columns or 'A1_Country' not in self.df.columns:
            self.logger.warning("Insufficient data for enforcement pattern analysis")
            return None

        # Aggregate enforcement data by country
        country_stats = self.df.groupby('A1_Country').agg({
            'A46_FineAmount_EUR': ['count', 'mean', 'median', 'sum'],
            'ID': 'count'
        }).round(2)

        country_stats.columns = ['Fine_Count', 'Mean_Fine', 'Median_Fine', 'Total_Fines', 'Total_Cases']
        country_stats = country_stats.reset_index()

        # Create scatter plot: Case count vs Average fine
        fig = px.scatter(
            country_stats,
            x='Total_Cases',
            y='Mean_Fine',
            size='Total_Fines',
            color='A1_Country',
            hover_data=['Fine_Count', 'Median_Fine'],
            title="GDPR Enforcement Patterns by Country",
            labels={
                'Total_Cases': 'Total Cases',
                'Mean_Fine': 'Average Fine Amount (EUR)',
                'Total_Fines': 'Total Fine Amount'
            }
        )

        # Add annotations for key insights
        if len(country_stats) > 0:
            max_cases_country = country_stats.loc[country_stats['Total_Cases'].idxmax()]
            max_fine_country = country_stats.loc[country_stats['Mean_Fine'].idxmax()]

            fig.add_annotation(
                x=max_cases_country['Total_Cases'],
                y=max_cases_country['Mean_Fine'],
                text=f"Most Cases:<br>{max_cases_country['A1_Country']}",
                showarrow=True,
                arrowhead=2
            )

        fig.update_layout(
            width=800,
            height=600,
            xaxis_title="Number of Cases",
            yaxis_title="Average Fine Amount (EUR)"
        )

        # Save chart
        output_path = self.output_dir / "enforcement_patterns.html"
        fig.write_html(str(output_path))
        self.charts_created.append("enforcement_patterns")
        self.logger.info("Created enforcement pattern analysis")
        return str(output_path)

    def create_significance_overview(self) -> str:
        """Create overview of statistical significance across relationships."""
        # Collect all significant relationships
        significant_relationships = []

        for category, relationships in self.bivariate_results.items():
            if category == 'metadata' or category == 'summary':
                continue

            for pair_key, result in relationships.items():
                if isinstance(result, dict) and 'p_value' in result:
                    p_value = result.get('p_value', 1)
                    if p_value < 0.05:  # Significant
                        effect_size = (
                            result.get('cramers_v', 0) if 'cramers_v' in result else
                            result.get('eta_squared', 0) if 'eta_squared' in result else
                            abs(result.get('pearson_r', 0)) if 'pearson_r' in result else 0
                        )

                        significant_relationships.append({
                            'Relationship': pair_key.replace('_vs_', ' vs '),
                            'Type': category.replace('_', '-').title(),
                            'P_Value': p_value,
                            'Effect_Size': effect_size,
                            'Interpretation': result.get('interpretation', ''),
                            'Neg_Log_P': -np.log10(p_value)  # For visualization
                        })

        if not significant_relationships:
            self.logger.warning("No significant relationships found")
            return None

        # Create DataFrame
        sig_df = pd.DataFrame(significant_relationships)

        # Create bubble plot
        fig = px.scatter(
            sig_df,
            x='Effect_Size',
            y='Neg_Log_P',
            size='Neg_Log_P',
            color='Type',
            hover_data=['Relationship', 'P_Value', 'Interpretation'],
            title="Statistical Significance Overview<br><sub>Effect Size vs -log₁₀(p-value)</sub>",
            labels={
                'Effect_Size': 'Effect Size',
                'Neg_Log_P': '-log₁₀(p-value)',
                'Type': 'Relationship Type'
            }
        )

        # Add significance threshold line
        fig.add_hline(y=-np.log10(0.05), line_dash="dash",
                     annotation_text="p = 0.05", annotation_position="right")

        # Add effect size thresholds
        fig.add_vline(x=0.1, line_dash="dot",
                     annotation_text="Small Effect", annotation_position="top")
        fig.add_vline(x=0.3, line_dash="dot",
                     annotation_text="Medium Effect", annotation_position="top")

        fig.update_layout(
            width=900,
            height=600,
            xaxis_title="Effect Size",
            yaxis_title="-log₁₀(p-value)"
        )

        # Save chart
        output_path = self.output_dir / "significance_overview.html"
        fig.write_html(str(output_path))
        self.charts_created.append("significance_overview")
        self.logger.info("Created significance overview")
        return str(output_path)

    def create_comprehensive_dashboard(self) -> str:
        """Create comprehensive bivariate analysis dashboard."""
        # Create main dashboard with key insights
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Relationship Strength Distribution",
                "P-Value Distribution",
                "Analysis Type Coverage",
                "Sample Size Distribution"
            ],
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "box"}]
            ]
        )

        # Collect all analysis results for overview
        all_results = []
        for category, relationships in self.bivariate_results.items():
            if category in ['metadata', 'summary']:
                continue
            for pair_key, result in relationships.items():
                if isinstance(result, dict):
                    all_results.append({
                        'category': category,
                        'p_value': result.get('p_value', result.get('kw_p_value', result.get('pearson_p_value', 1))),
                        'effect_size': (
                            result.get('cramers_v', 0) if 'cramers_v' in result else
                            result.get('eta_squared', 0) if 'eta_squared' in result else
                            abs(result.get('pearson_r', 0)) if 'pearson_r' in result else 0
                        ),
                        'sample_size': result.get('sample_size', 0)
                    })

        if not all_results:
            self.logger.warning("No results available for dashboard")
            return None

        results_df = pd.DataFrame(all_results)

        # 1. Effect size distribution
        fig.add_trace(
            go.Histogram(x=results_df['effect_size'], name="Effect Sizes", nbinsx=20),
            row=1, col=1
        )

        # 2. P-value distribution
        fig.add_trace(
            go.Histogram(x=results_df['p_value'], name="P-Values", nbinsx=20),
            row=1, col=2
        )

        # 3. Analysis type coverage
        type_counts = results_df['category'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values, name="Analysis Types"),
            row=2, col=1
        )

        # 4. Sample size distribution by category
        for category in results_df['category'].unique():
            cat_data = results_df[results_df['category'] == category]['sample_size']
            fig.add_trace(
                go.Box(y=cat_data, name=category.replace('_', ' ').title()),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title="Bivariate Analysis Dashboard",
            height=800,
            width=1200,
            showlegend=False
        )

        # Save dashboard
        output_path = self.output_dir / "bivariate_dashboard.html"
        fig.write_html(str(output_path))
        self.charts_created.append("bivariate_dashboard")
        self.logger.info("Created comprehensive dashboard")
        return str(output_path)

    def generate_all_visualizations(self) -> Dict[str, Any]:
        """Generate all bivariate visualizations."""
        self.logger.info("Generating comprehensive bivariate visualizations...")

        visualization_paths = {}

        # Create individual visualizations
        viz_methods = [
            ('correlation_heatmap', self.create_correlation_heatmap),
            ('fine_amount_analysis', self.create_fine_amount_analysis),
            ('categorical_relationships', self.create_categorical_relationships_grid),
            ('enforcement_patterns', self.create_enforcement_pattern_analysis),
            ('significance_overview', self.create_significance_overview),
            ('dashboard', self.create_comprehensive_dashboard)
        ]

        for viz_name, viz_method in viz_methods:
            try:
                result = viz_method()
                if result:
                    visualization_paths[viz_name] = result
                    self.logger.info(f"✓ Created {viz_name}")
                else:
                    self.logger.warning(f"⚠ Skipped {viz_name} (insufficient data)")
            except Exception as e:
                self.logger.error(f"✗ Failed to create {viz_name}: {e}")

        # Save summary
        summary = {
            'total_visualizations': len(self.charts_created),
            'charts_created': self.charts_created,
            'visualization_paths': visualization_paths,
            'generation_timestamp': datetime.now().isoformat()
        }

        summary_path = self.output_dir / "visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Generated {len(self.charts_created)} bivariate visualizations")
        return summary

def main():
    """Main execution function."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Bivariate Visualization Engine")
    print("=" * 40)

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

    # Initialize visualization engine
    viz_engine = BivariateVisualizationEngine(data_path, bivariate_path)

    # Generate all visualizations
    print("🎨 Generating bivariate visualizations...")
    summary = viz_engine.generate_all_visualizations()

    print("\n✅ Bivariate Visualization Complete!")
    print(f"📊 Generated {summary['total_visualizations']} visualizations")
    print(f"📁 Visualizations saved to: bivariate_visualizations/")
    print("🌐 Open the HTML files in your browser for interactive exploration")

if __name__ == "__main__":
    main()