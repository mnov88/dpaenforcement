#!/usr/bin/env python3
"""
Interactive Visualization Engine
================================

Comprehensive visualization system for univariate analysis results with
interactive plotly charts, automated chart selection, and export capabilities.

Author: Enhanced Data Analysis Pipeline
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class VisualizationEngine:
    """Interactive visualization engine for univariate analysis results."""

    def __init__(self, df: pd.DataFrame, analysis_results: Dict[str, Any]):
        """Initialize visualization engine with data and analysis results."""
        self.df = df

        # Access the complete analysis results structure
        if 'complete' in analysis_results:
            self.analysis_results = analysis_results['complete']
        else:
            self.analysis_results = analysis_results

        self.charts = {}
        self.chart_metadata = {}

        # Visualization settings
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        self.width = 800
        self.height = 600

        # Output directory
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

        print(f"Initialized VisualizationEngine with {len(df)} rows and {len(df.columns)} columns")
        print(f"Using analysis results with {len(self.analysis_results)} sections")

    def create_categorical_charts(self) -> Dict[str, go.Figure]:
        """Create comprehensive charts for categorical variables."""
        print("📊 Creating categorical variable charts...")

        categorical_charts = {}

        if 'categorical_analysis' not in self.analysis_results:
            return categorical_charts

        for col, analysis in self.analysis_results['categorical_analysis'].items():
            print(f"  Creating charts for {col}...")

            freq_dist = analysis['frequency_distribution']
            perc_dist = analysis['percentage_distribution']

            # Sort by frequency for better visualization
            sorted_categories = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

            # Limit to top categories if too many
            if len(sorted_categories) > 15:
                top_categories = sorted_categories[:12]
                other_count = sum([count for _, count in sorted_categories[12:]])
                top_categories.append(('Others', other_count))
                categories, counts = zip(*top_categories)
            else:
                categories, counts = zip(*sorted_categories)

            # Create main frequency chart
            fig = go.Figure()

            # Bar chart
            fig.add_trace(go.Bar(
                x=list(categories),
                y=list(counts),
                name=f'{col} Frequency',
                marker_color=self.color_palette[0],
                text=[f'{count}<br>({count/sum(counts)*100:.1f}%)' for count in counts],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
            ))

            fig.update_layout(
                title=f'Frequency Distribution: {col}',
                xaxis_title='Categories',
                yaxis_title='Frequency',
                template=self.template,
                width=self.width,
                height=self.height,
                xaxis_tickangle=-45 if len(categories) > 5 else 0
            )

            # Add insights as annotation
            insights_text = '<br>'.join([f"• {insight}" for insight in analysis.get('insights', [])])
            if insights_text:
                fig.add_annotation(
                    text=insights_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )

            categorical_charts[f'{col}_frequency'] = fig

            # Create pie chart for variables with reasonable number of categories
            if len(categories) <= 8:
                pie_fig = go.Figure(data=[go.Pie(
                    labels=list(categories),
                    values=list(counts),
                    hole=0.3,
                    marker_colors=self.color_palette[:len(categories)]
                )])

                pie_fig.update_layout(
                    title=f'Distribution: {col}',
                    template=self.template,
                    width=self.width,
                    height=self.height
                )

                categorical_charts[f'{col}_pie'] = pie_fig

        self.charts['categorical'] = categorical_charts
        print(f"  Created {len(categorical_charts)} categorical charts")
        return categorical_charts

    def create_continuous_charts(self) -> Dict[str, go.Figure]:
        """Create comprehensive charts for continuous variables."""
        print("📊 Creating continuous variable charts...")

        continuous_charts = {}

        if 'continuous_analysis' not in self.analysis_results:
            return continuous_charts

        for col, analysis in self.analysis_results['continuous_analysis'].items():
            print(f"  Creating charts for {col}...")

            data = self.df[col].dropna()
            if len(data) == 0:
                continue

            desc_stats = analysis['descriptive_statistics']

            # 1. Histogram with distribution overlay
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram with KDE', 'Box Plot', 'Q-Q Plot', 'Summary Statistics'),
                specs=[[{"secondary_y": True}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )

            # Histogram
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=min(30, len(data)//3),
                name='Frequency',
                marker_color=self.color_palette[0],
                opacity=0.7
            ), row=1, col=1)

            # Add KDE curve
            from scipy.stats import gaussian_kde
            if len(data) > 2:
                try:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    kde_values = kde(x_range)
                    # Scale KDE to match histogram
                    kde_scaled = kde_values * len(data) * (data.max() - data.min()) / min(30, len(data)//3)

                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_scaled,
                        mode='lines',
                        name='Density',
                        line=dict(color='red', width=2)
                    ), row=1, col=1, secondary_y=True)
                except:
                    pass

            # Box plot
            fig.add_trace(go.Box(
                y=data,
                name=col,
                marker_color=self.color_palette[1],
                boxpoints='outliers'
            ), row=1, col=2)

            # Q-Q plot for normality
            from scipy import stats
            try:
                qq_sample = np.random.choice(data, min(100, len(data)), replace=False)
                qq_theoretical, qq_sample_sorted = stats.probplot(qq_sample, dist="norm")

                fig.add_trace(go.Scatter(
                    x=qq_theoretical[0],
                    y=qq_theoretical[1],
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color=self.color_palette[2])
                ), row=2, col=1)

                # Add reference line
                fig.add_trace(go.Scatter(
                    x=qq_theoretical[0],
                    y=qq_theoretical[0] * np.std(qq_sample_sorted) + np.mean(qq_sample_sorted),
                    mode='lines',
                    name='Normal Reference',
                    line=dict(color='red', dash='dash')
                ), row=2, col=1)
            except:
                pass

            # Summary statistics table
            stats_data = [
                ['Statistic', 'Value'],
                ['Count', f"{desc_stats['count']:,}"],
                ['Mean', f"{desc_stats['mean']:,.2f}"],
                ['Median', f"{desc_stats['median']:,.2f}"],
                ['Std Dev', f"{desc_stats['std']:,.2f}"],
                ['Min', f"{desc_stats['min']:,.2f}"],
                ['Max', f"{desc_stats['max']:,.2f}"],
                ['Skewness', f"{desc_stats['skewness']:.3f}"],
                ['Kurtosis', f"{desc_stats['kurtosis']:.3f}"]
            ]

            fig.add_trace(go.Table(
                header=dict(values=stats_data[0], fill_color='lightblue'),
                cells=dict(values=list(zip(*stats_data[1:])), fill_color='white')
            ), row=2, col=2)

            fig.update_layout(
                title=f'Comprehensive Analysis: {col}',
                template=self.template,
                width=self.width * 1.5,
                height=self.height * 1.2,
                showlegend=True
            )

            continuous_charts[f'{col}_comprehensive'] = fig

            # Create separate detailed histogram for fine amounts
            if 'fine' in col.lower() or 'amount' in col.lower():
                fine_fig = self._create_fine_analysis_chart(data, col, analysis)
                continuous_charts[f'{col}_fine_analysis'] = fine_fig

        self.charts['continuous'] = continuous_charts
        print(f"  Created {len(continuous_charts)} continuous charts")
        return continuous_charts

    def _create_fine_analysis_chart(self, data: pd.Series, col: str, analysis: Dict) -> go.Figure:
        """Create specialized chart for fine amount analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Linear Scale Distribution', 'Log Scale Distribution',
                          'Outlier Analysis', 'Percentile Breakdown'),
            specs=[[{}, {}], [{}, {"type": "table"}]]
        )

        # Linear scale histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=30,
            name='Linear Scale',
            marker_color=self.color_palette[0]
        ), row=1, col=1)

        # Log scale histogram (for non-zero values)
        non_zero_data = data[data > 0]
        if len(non_zero_data) > 0:
            fig.add_trace(go.Histogram(
                x=np.log10(non_zero_data),
                nbinsx=20,
                name='Log10 Scale',
                marker_color=self.color_palette[1]
            ), row=1, col=2)

        # Outlier analysis
        outliers = analysis.get('outliers', {})
        extreme_outliers = outliers.get('extreme_outliers', [])

        # All data points
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data,
            mode='markers',
            name='All Values',
            marker=dict(color='blue', size=4),
            opacity=0.6
        ), row=2, col=1)

        # Highlight outliers
        if extreme_outliers:
            outlier_indices = [i for i, val in enumerate(data) if val in extreme_outliers]
            fig.add_trace(go.Scatter(
                x=outlier_indices,
                y=[data.iloc[i] for i in outlier_indices],
                mode='markers',
                name='Extreme Outliers',
                marker=dict(color='red', size=8, symbol='diamond')
            ), row=2, col=1)

        # Percentile table
        percentiles = analysis.get('percentiles', {})
        perc_data = [
            ['Percentile', 'Value (EUR)'],
            ['5th', f"€{percentiles.get('p5', 0):,.0f}"],
            ['25th', f"€{percentiles.get('p25', 0):,.0f}"],
            ['50th (Median)', f"€{percentiles.get('p50', 0):,.0f}"],
            ['75th', f"€{percentiles.get('p75', 0):,.0f}"],
            ['95th', f"€{percentiles.get('p95', 0):,.0f}"]
        ]

        fig.add_trace(go.Table(
            header=dict(values=perc_data[0], fill_color='lightgreen'),
            cells=dict(values=list(zip(*perc_data[1:])), fill_color='white')
        ), row=2, col=2)

        fig.update_layout(
            title=f'Fine Amount Analysis: {col}',
            template=self.template,
            width=self.width * 1.5,
            height=self.height * 1.2
        )

        return fig

    def create_binary_charts(self) -> Dict[str, go.Figure]:
        """Create charts for binary variables."""
        print("📊 Creating binary variable charts...")

        binary_charts = {}

        if 'binary_analysis' not in self.analysis_results:
            return binary_charts

        # Aggregate binary variables for overview
        binary_data = []
        for col, analysis in self.analysis_results['binary_analysis'].items():
            binary_data.append({
                'Variable': col.replace('_', ' ').title(),
                'Positive_Rate': analysis['positive_percentage'],
                'Total_Cases': analysis['total_observations'],
                'Positive_Count': analysis['positive_count'],
                'Is_Rare': analysis['is_rare_event']
            })

        binary_df = pd.DataFrame(binary_data)

        # Overview chart: Positive rates across all binary variables
        fig = go.Figure()

        colors = ['red' if rare else 'blue' for rare in binary_df['Is_Rare']]

        fig.add_trace(go.Bar(
            x=binary_df['Variable'],
            y=binary_df['Positive_Rate'],
            marker_color=colors,
            text=[f"{rate:.1f}%<br>({count}/{total})"
                  for rate, count, total in zip(binary_df['Positive_Rate'],
                                              binary_df['Positive_Count'],
                                              binary_df['Total_Cases'])],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Positive Rate: %{y:.1f}%<br>%{text}<extra></extra>'
        ))

        # Add 10% and 90% reference lines
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                     annotation_text="10% (Rare Event Threshold)")
        fig.add_hline(y=90, line_dash="dash", line_color="red",
                     annotation_text="90% (Common Event Threshold)")

        fig.update_layout(
            title='Binary Variables: Positive Event Rates',
            xaxis_title='Variables',
            yaxis_title='Positive Rate (%)',
            template=self.template,
            width=self.width * 1.5,
            height=self.height,
            xaxis_tickangle=-45
        )

        binary_charts['overview'] = fig

        # Create detailed charts for interesting binary variables
        for col, analysis in self.analysis_results['binary_analysis'].items():
            if not analysis['is_rare_event']:  # Focus on balanced variables
                fig = self._create_binary_detail_chart(col, analysis)
                binary_charts[f'{col}_detail'] = fig

        self.charts['binary'] = binary_charts
        print(f"  Created {len(binary_charts)} binary charts")
        return binary_charts

    def _create_binary_detail_chart(self, col: str, analysis: Dict) -> go.Figure:
        """Create detailed chart for a single binary variable."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{col} Distribution', 'Confidence Interval'),
            specs=[[{"type": "pie"}, {}]]
        )

        # Pie chart
        labels = ['Positive', 'Negative']
        values = [analysis['positive_count'], analysis['negative_count']]

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=['green', 'lightgray']
        ), row=1, col=1)

        # Confidence interval visualization
        ci_lower, ci_upper = analysis['confidence_interval_95']
        point_estimate = analysis['positive_rate']

        fig.add_trace(go.Scatter(
            x=[point_estimate],
            y=[1],
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='Point Estimate',
            showlegend=True
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[1, 1],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            name='95% CI',
            showlegend=True
        ), row=1, col=2)

        fig.update_layout(
            title=f'Binary Variable Analysis: {col}',
            template=self.template,
            width=self.width * 1.2,
            height=self.height
        )

        # Update x-axis for CI plot
        fig.update_xaxes(title_text="Positive Rate", range=[0, 1], row=1, col=2)
        fig.update_yaxes(showticklabels=False, range=[0.5, 1.5], row=1, col=2)

        return fig

    def create_temporal_charts(self) -> Dict[str, go.Figure]:
        """Create charts for temporal analysis."""
        print("📊 Creating temporal charts...")

        temporal_charts = {}

        if 'temporal_analysis' not in self.analysis_results:
            return temporal_charts

        for col, analysis in self.analysis_results['temporal_analysis'].items():
            print(f"  Creating temporal charts for {col}...")

            # Multi-panel temporal analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Yearly Trend', 'Monthly Distribution',
                              'Quarterly Pattern', 'Weekday Pattern')
            )

            # Yearly trend
            yearly_data = analysis['yearly_distribution']
            years = list(yearly_data.keys())
            counts = list(yearly_data.values())

            fig.add_trace(go.Scatter(
                x=years,
                y=counts,
                mode='lines+markers',
                name='Yearly Trend',
                line=dict(color=self.color_palette[0], width=3),
                marker=dict(size=8)
            ), row=1, col=1)

            # Monthly distribution
            monthly_data = analysis['monthly_distribution']
            months = [f"Month {m}" for m in sorted(monthly_data.keys())]
            month_counts = [monthly_data[m] for m in sorted(monthly_data.keys())]

            fig.add_trace(go.Bar(
                x=months,
                y=month_counts,
                name='Monthly',
                marker_color=self.color_palette[1]
            ), row=1, col=2)

            # Quarterly pattern
            quarterly_data = analysis['quarterly_distribution']
            quarters = [f"Q{q}" for q in sorted(quarterly_data.keys())]
            quarter_counts = [quarterly_data[q] for q in sorted(quarterly_data.keys())]

            fig.add_trace(go.Bar(
                x=quarters,
                y=quarter_counts,
                name='Quarterly',
                marker_color=self.color_palette[2]
            ), row=2, col=1)

            # Weekday pattern
            weekday_data = analysis['weekday_distribution']
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekdays = [day for day in weekday_order if day in weekday_data]
            weekday_counts = [weekday_data[day] for day in weekdays]

            fig.add_trace(go.Bar(
                x=weekdays,
                y=weekday_counts,
                name='Weekday',
                marker_color=self.color_palette[3]
            ), row=2, col=2)

            fig.update_layout(
                title=f'Temporal Analysis: {col}',
                template=self.template,
                width=self.width * 1.5,
                height=self.height * 1.2,
                showlegend=False
            )

            temporal_charts[f'{col}_temporal'] = fig

        self.charts['temporal'] = temporal_charts
        print(f"  Created {len(temporal_charts)} temporal charts")
        return temporal_charts

    def create_overview_dashboard(self) -> go.Figure:
        """Create high-level overview dashboard."""
        print("📊 Creating overview dashboard...")

        # Get key metrics
        exec_summary = self.analysis_results.get('executive_summary', {})
        dataset_overview = exec_summary.get('dataset_overview', {})

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dataset Overview', 'Data Quality', 'Variable Types', 'Key Insights'),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "table"}]]
        )

        # Dataset overview table
        overview_data = [
            ['Metric', 'Value'],
            ['Total Rows', f"{dataset_overview.get('total_rows', 0):,}"],
            ['Total Columns', f"{dataset_overview.get('total_columns', 0):,}"],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['Data Span', 'Norwegian GDPR Cases']
        ]

        fig.add_trace(go.Table(
            header=dict(values=overview_data[0], fill_color='lightblue'),
            cells=dict(values=list(zip(*overview_data[1:])), fill_color='white')
        ), row=1, col=1)

        # Data quality metrics (simplified)
        quality_metrics = []
        quality_values = []

        for col, profile in self.analysis_results.get('variable_profiles', {}).items():
            if profile.get('null_percentage', 0) < 20:
                quality_metrics.append('Good Quality')
            else:
                quality_metrics.append('Needs Attention')

        quality_counts = pd.Series(quality_metrics).value_counts()

        fig.add_trace(go.Bar(
            x=list(quality_counts.index),
            y=list(quality_counts.values),
            marker_color=['green', 'orange'],
            name='Data Quality'
        ), row=1, col=2)

        # Variable types pie chart
        var_breakdown = dataset_overview.get('variable_breakdown', {})
        var_types = list(var_breakdown.keys())
        var_counts = list(var_breakdown.values())

        fig.add_trace(go.Pie(
            labels=var_types,
            values=var_counts,
            hole=0.3,
            marker_colors=self.color_palette[:len(var_types)]
        ), row=2, col=1)

        # Key insights table
        key_findings = exec_summary.get('key_findings', [])[:6]  # Top 6 findings
        insights_data = [['Key Insight']] + [[finding] for finding in key_findings]

        fig.add_trace(go.Table(
            header=dict(values=insights_data[0], fill_color='lightgreen'),
            cells=dict(values=list(zip(*insights_data[1:])), fill_color='white')
        ), row=2, col=2)

        fig.update_layout(
            title='GDPR Enforcement Data: Analysis Overview',
            template=self.template,
            width=self.width * 1.8,
            height=self.height * 1.5,
            showlegend=False
        )

        return fig

    def export_all_charts(self, format: str = 'html') -> Dict[str, str]:
        """Export all charts to specified format."""
        print(f"📁 Exporting all charts to {format.upper()} format...")

        exported_files = {}

        # Create all charts first
        self.create_categorical_charts()
        self.create_continuous_charts()
        self.create_binary_charts()
        self.create_temporal_charts()

        # Create overview dashboard
        overview_chart = self.create_overview_dashboard()
        self.charts['overview'] = {'dashboard': overview_chart}

        # Export each chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for category, charts in self.charts.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)

            for chart_name, chart in charts.items():
                filename = f"{chart_name}_{timestamp}"

                if format.lower() == 'html':
                    filepath = category_dir / f"{filename}.html"
                    chart.write_html(str(filepath))
                elif format.lower() == 'png':
                    filepath = category_dir / f"{filename}.png"
                    chart.write_image(str(filepath))
                elif format.lower() == 'pdf':
                    filepath = category_dir / f"{filename}.pdf"
                    chart.write_image(str(filepath))

                exported_files[f"{category}_{chart_name}"] = str(filepath)

        print(f"  Exported {len(exported_files)} charts to {self.output_dir}")
        return exported_files

    def generate_chart_summary(self) -> Dict[str, Any]:
        """Generate summary of all created charts."""
        summary = {
            'total_charts': sum(len(charts) for charts in self.charts.values()),
            'chart_categories': list(self.charts.keys()),
            'charts_by_category': {cat: list(charts.keys()) for cat, charts in self.charts.items()},
            'generation_timestamp': datetime.now().isoformat()
        }

        return summary


def main():
    """Main execution function."""
    print("Interactive Visualization Engine")
    print("=" * 50)

    # Load analysis results
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

    # Initialize visualization engine
    viz_engine = VisualizationEngine(df, analysis_results)

    # Generate all visualizations
    print("\n🎨 Generating interactive visualizations...")

    exported_files = viz_engine.export_all_charts('html')

    # Generate summary
    chart_summary = viz_engine.generate_chart_summary()

    print(f"\n✅ Visualization Complete!")
    print(f"📊 Generated {chart_summary['total_charts']} interactive charts")
    print(f"📁 Charts exported to: visualizations/")
    print(f"🌐 Open the HTML files in your browser for interactive exploration")

    # Save chart summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'chart_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(chart_summary, f, indent=2)

    print(f"📋 Chart summary saved to: {summary_file}")


if __name__ == "__main__":
    main()