"""
Module 2: Advanced EDA
Scatter plots, correlation heatmaps, interaction effects, ANOVA testing,
and performance category creation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedEDA:
    def __init__(self, df):
        """Initialize EDA with dataframe."""
        self.df = df
        self.target = 'Exam_Score'
        self.performance_categories = None
        
    def create_performance_categories(self):
        """Create performance categories (low/medium/high risk)."""
        print("\nCreating performance categories...")
        
        # Define thresholds based on percentiles
        low_threshold = self.df[self.target].quantile(0.33)
        high_threshold = self.df[self.target].quantile(0.67)
        
        def categorize_score(score):
            if score < low_threshold:
                return 'Low Performance (High Risk)'
            elif score < high_threshold:
                return 'Medium Performance (Medium Risk)'
            else:
                return 'High Performance (Low Risk)'
        
        self.df['Performance_Category'] = self.df[self.target].apply(categorize_score)
        self.performance_categories = self.df['Performance_Category'].value_counts()
        
        print("Performance categories created:")
        print(self.performance_categories)
        return self.df
    
    def create_scatter_plots(self, features=None, save_path='scatter_plots.html'):
        """Create interactive scatter plots."""
        print("\nCreating scatter plots...")
        
        if features is None:
            # Select top numerical features
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target in numerical_cols:
                numerical_cols.remove(self.target)
            if 'Performance_Category' in numerical_cols:
                numerical_cols.remove('Performance_Category')
            features = numerical_cols[:6]  # Top 6 features
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=features,
            specs=[[{"secondary_y": False}]*3]*2
        )
        
        for idx, feature in enumerate(features):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[feature],
                    y=self.df[self.target],
                    mode='markers',
                    marker=dict(size=5, opacity=0.6),
                    name=feature,
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text=self.target, row=row, col=col)
        
        fig.update_layout(
            title_text="Scatter Plots: Features vs Exam Score",
            height=800,
            showlegend=False
        )
        
        fig.write_html(save_path)
        print(f"Scatter plots saved to '{save_path}'")
        return fig
    
    def create_correlation_heatmap(self, save_path='correlation_heatmap.html'):
        """Create correlation heatmap."""
        print("\nCreating correlation heatmap...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self.df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            width=1000,
            height=800
        )
        
        fig.write_html(save_path)
        print(f"Correlation heatmap saved to '{save_path}'")
        return fig
    
    def get_top_correlations(self, n=10):
        """Get top feature correlations with target."""
        print(f"\nGetting top {n} correlations with target...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numerical_cols:
            numerical_cols.remove(self.target)
        
        correlations = {}
        for col in numerical_cols:
            corr = self.df[col].corr(self.df[self.target])
            correlations[col] = abs(corr)
        
        top_corr = pd.Series(correlations).sort_values(ascending=False).head(n)
        
        print("Top correlations with target:")
        print(top_corr)
        
        # Save to CSV
        top_corr_df = pd.DataFrame({
            'Feature': top_corr.index,
            'Absolute_Correlation': top_corr.values
        })
        top_corr_df.to_csv('top_correlations.csv', index=False)
        
        return top_corr
    
    def visualize_interaction_effects(self, feature1, feature2, save_path='interaction_effects.html'):
        """Visualize interaction effects between two features."""
        print(f"\nVisualizing interaction effects: {feature1} x {feature2}...")
        
        fig = px.scatter_3d(
            self.df,
            x=feature1,
            y=feature2,
            z=self.target,
            color='Performance_Category' if 'Performance_Category' in self.df.columns else self.target,
            title=f"Interaction Effect: {feature1} x {feature2}",
            labels={feature1: feature1, feature2: feature2, self.target: self.target}
        )
        
        fig.write_html(save_path)
        print(f"Interaction effect visualization saved to '{save_path}'")
        return fig
    
    def detect_nonlinear_trends(self):
        """Detect non-linear trends using polynomial regression."""
        print("\nDetecting non-linear trends...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numerical_cols:
            numerical_cols.remove(self.target)
        
        nonlinear_results = {}
        
        for col in numerical_cols[:10]:  # Limit to first 10 for performance
            x = self.df[col].values
            y = self.df[self.target].values
            
            # Remove NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 10:
                continue
            
            # Linear correlation
            linear_corr = np.corrcoef(x_clean, y_clean)[0, 1]
            
            # Polynomial fit (degree 2)
            try:
                coeffs = np.polyfit(x_clean, y_clean, 2)
                y_poly = np.polyval(coeffs, x_clean)
                poly_corr = np.corrcoef(y_clean, y_poly)[0, 1]
                
                nonlinear_results[col] = {
                    'linear_correlation': linear_corr,
                    'polynomial_correlation': poly_corr,
                    'nonlinear_improvement': poly_corr - abs(linear_corr)
                }
            except:
                continue
        
        nonlinear_df = pd.DataFrame(nonlinear_results).T
        nonlinear_df = nonlinear_df.sort_values('nonlinear_improvement', ascending=False)
        nonlinear_df.to_csv('nonlinear_trends.csv')
        
        print("Non-linear trend analysis saved to 'nonlinear_trends.csv'")
        return nonlinear_df
    
    def perform_anova_test(self):
        """Perform ANOVA testing for categorical features."""
        print("\nPerforming ANOVA tests...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if 'Performance_Category' in categorical_cols:
            categorical_cols.remove('Performance_Category')

        # If the dataframe has no categorical columns (common after encoding),
        # skip gracefully instead of crashing.
        if len(categorical_cols) == 0:
            print("No categorical features available for ANOVA (object dtype). Skipping ANOVA.")
            empty = pd.DataFrame(columns=['F_statistic', 'p_value', 'significant'])
            empty.to_csv('anova_results.csv', index=False)
            return empty
        
        anova_results = {}
        
        for col in categorical_cols:
            groups = []
            for category in self.df[col].unique():
                if pd.notna(category):
                    group_data = self.df[self.df[col] == category][self.target].values
                    if len(group_data) > 0:
                        groups.append(group_data)
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = f_oneway(*groups)
                    anova_results[col] = {
                        'F_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    continue
        
        anova_df = pd.DataFrame(anova_results).T

        # If nothing was computed, write an empty file and return.
        if anova_df.empty:
            print("ANOVA produced no results (insufficient groups). Saving empty anova_results.csv.")
            empty = pd.DataFrame(columns=['F_statistic', 'p_value', 'significant'])
            empty.to_csv('anova_results.csv', index=False)
            return empty

        anova_df = anova_df.sort_values('p_value')
        anova_df.to_csv('anova_results.csv')
        
        print("ANOVA results saved to 'anova_results.csv'")
        print(f"\nSignificant features (p < 0.05): {len(anova_df[anova_df['significant']])}")
        return anova_df
    
    def create_distribution_plots(self, save_path='distributions.html'):
        """Create distribution plots for key features."""
        print("\nCreating distribution plots...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numerical_cols:
            numerical_cols.remove(self.target)
        
        top_features = self.get_top_correlations(n=6).index.tolist()
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=top_features
        )
        
        for idx, feature in enumerate(top_features):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.df[feature],
                    nbinsx=30,
                    name=feature,
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text="Frequency", row=row, col=col)
        
        fig.update_layout(
            title_text="Distribution of Top Features",
            height=800
        )
        
        fig.write_html(save_path)
        print(f"Distribution plots saved to '{save_path}'")
        return fig
    
    def generate_eda_report(self):
        """Generate comprehensive EDA report."""
        print("\n" + "="*50)
        print("Generating EDA Report...")
        print("="*50)
        
        # Create performance categories
        self.create_performance_categories()
        
        # Scatter plots
        self.create_scatter_plots()
        
        # Correlation heatmap
        self.create_correlation_heatmap()
        
        # Top correlations
        self.get_top_correlations()
        
        # Non-linear trends
        self.detect_nonlinear_trends()
        
        # ANOVA tests
        self.perform_anova_test()
        
        # Distribution plots
        self.create_distribution_plots()
        
        print("\n" + "="*50)
        print("Module 2: Advanced EDA Complete!")
        print("="*50)
        
        return self.df


if __name__ == "__main__":
    # Load preprocessed data
    try:
        df = pd.read_csv('preprocessed_data.csv')
    except:
        print("Preprocessed data not found. Loading raw data...")
        df = pd.read_csv('StudentPerformanceFactors.csv')
    
    # Initialize EDA
    eda = AdvancedEDA(df)
    
    # Generate EDA report
    df_with_categories = eda.generate_eda_report()
    
    # Save dataframe with performance categories
    df_with_categories.to_csv('data_with_performance_categories.csv', index=False)
