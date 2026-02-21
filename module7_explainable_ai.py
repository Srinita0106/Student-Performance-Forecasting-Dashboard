"""
Module 7: Explainable AI & Counterfactuals
SHAP analysis, global + local importance, counterfactual scenarios,
and risk factor prioritization.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ExplainableAI:
    def __init__(self, model, X_train, X_test, feature_names=None):
        """Initialize explainable AI system."""
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names if feature_names else X_train.columns.tolist()
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self, explainer_type='tree'):
        """Create SHAP explainer."""
        print("\nCreating SHAP explainer...")
        
        if explainer_type == 'tree':
            # Tree-based explainer (for XGBoost, LightGBM, CatBoost)
            # NOTE: Some newer XGBoost versions are incompatible with older SHAP
            # releases and can raise errors such as:
            # "ValueError: could not convert string to float: '[6.721513E1]'".
            # We try the fast tree explainer first, and if it fails, fall back
            # to a model-agnostic explainer so the module still completes.
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"TreeExplainer failed with error: {e}")
                print("Falling back to model-agnostic SHAP Explainer (permutation).")
                background = (
                    self.X_train.sample(
                        n=min(200, len(self.X_train)),
                        random_state=42
                    )
                    if hasattr(self.X_train, "sample") and len(self.X_train) > 0
                    else self.X_train
                )
                self.explainer = shap.Explainer(
                    self.model.predict,
                    background,
                    algorithm="permutation"
                )
        elif explainer_type == 'kernel':
            # Kernel explainer (for any model)
            self.explainer = shap.KernelExplainer(self.model.predict, self.X_train[:100])
        elif explainer_type == 'linear':
            # Linear explainer (for linear models)
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        print(f"SHAP explainer created: {explainer_type}")
        return self.explainer
    
    def calculate_shap_values(self, X=None):
        """Calculate SHAP values."""
        print("\nCalculating SHAP values...")
        
        if X is None:
            X = self.X_test
        
        if self.explainer is None:
            self.create_shap_explainer()
        
        self.shap_values = self.explainer.shap_values(X)
        
        print(f"SHAP values calculated for {len(X)} samples")
        return self.shap_values
    
    def get_global_importance(self):
        """Get global feature importance."""
        print("\nCalculating global feature importance...")
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Mean absolute SHAP values
        if isinstance(self.shap_values, list):
            shap_values_array = np.array(self.shap_values)
        else:
            shap_values_array = self.shap_values
        
        global_importance = np.abs(shap_values_array).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names[:len(global_importance)],
            'Importance': global_importance
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv('global_feature_importance.csv', index=False)
        print("Global feature importance saved to 'global_feature_importance.csv'")
        
        return importance_df
    
    def get_local_importance(self, instance_idx):
        """Get local feature importance for a specific instance."""
        print(f"\nCalculating local importance for instance {instance_idx}...")
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if isinstance(self.shap_values, list):
            shap_values_array = np.array(self.shap_values)
        else:
            shap_values_array = self.shap_values
        
        local_shap = shap_values_array[instance_idx]
        
        local_importance = pd.DataFrame({
            'Feature': self.feature_names[:len(local_shap)],
            'SHAP_Value': local_shap,
            'Absolute_Impact': np.abs(local_shap)
        }).sort_values('Absolute_Impact', ascending=False)
        
        return local_importance
    
    def visualize_global_importance(self, top_n=20, save_path='global_importance.html'):
        """Visualize global feature importance."""
        print("\nCreating global importance visualization...")
        
        importance_df = self.get_global_importance()
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(data=go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker=dict(color=top_features['Importance'], 
                       colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=f"Global Feature Importance (Top {top_n})",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Feature",
            height=max(400, top_n * 30)
        )
        
        fig.write_html(save_path)
        print(f"Global importance visualization saved to '{save_path}'")
        return fig
    
    def visualize_local_importance(self, instance_idx, save_path='local_importance.html'):
        """Visualize local feature importance."""
        print(f"\nCreating local importance visualization for instance {instance_idx}...")
        
        local_importance = self.get_local_importance(instance_idx)
        
        fig = go.Figure()
        
        # Positive contributions
        positive = local_importance[local_importance['SHAP_Value'] > 0]
        fig.add_trace(go.Bar(
            x=positive['SHAP_Value'],
            y=positive['Feature'],
            orientation='h',
            name='Positive Impact',
            marker_color='green'
        ))
        
        # Negative contributions
        negative = local_importance[local_importance['SHAP_Value'] < 0]
        fig.add_trace(go.Bar(
            x=negative['SHAP_Value'],
            y=negative['Feature'],
            orientation='h',
            name='Negative Impact',
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f"Local Feature Importance - Instance {instance_idx}",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            height=max(400, len(local_importance) * 30)
        )
        
        fig.write_html(save_path)
        print(f"Local importance visualization saved to '{save_path}'")
        return fig
    
    def create_waterfall_plot(self, instance_idx, save_path='waterfall_plot.html'):
        """Create SHAP waterfall plot."""
        print(f"\nCreating waterfall plot for instance {instance_idx}...")
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if isinstance(self.shap_values, list):
            shap_values_array = np.array(self.shap_values)
        else:
            shap_values_array = self.shap_values
        
        local_shap = shap_values_array[instance_idx]
        # Some SHAP explainers (e.g., TreeExplainer) expose `expected_value`,
        # while others (e.g., PermutationExplainer) may not. Fall back to the
        # model's mean prediction when `expected_value` is unavailable.
        base_value = getattr(self.explainer, "expected_value", None)
        if base_value is None:
            base_value = float(self.model.predict(self.X_train).mean())
        else:
            # Handle array-like expected values
            try:
                base_value = float(np.mean(base_value))
            except Exception:
                base_value = float(self.model.predict(self.X_train).mean())
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(local_shap))[::-1]
        
        # Create waterfall
        features = [self.feature_names[i] for i in sorted_indices[:15]]
        values = local_shap[sorted_indices[:15]]
        
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(features) + ["total"],
            x=["Base Value"] + features + ["Prediction"],
            textposition="outside",
            text=[f"{base_value:.2f}"] + [f"{v:.2f}" for v in values] + 
                 [f"{base_value + sum(values):.2f}"],
            y=[base_value] + list(values) + [0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot - Instance {instance_idx}",
            showlegend=False,
            height=600
        )
        
        fig.write_html(save_path)
        print(f"Waterfall plot saved to '{save_path}'")
        return fig
    
    def generate_counterfactuals(self, instance_idx, target_score=None, n_counterfactuals=5):
        """Generate counterfactual scenarios."""
        print(f"\nGenerating counterfactuals for instance {instance_idx}...")
        
        instance = self.X_test.iloc[instance_idx].copy()
        current_prediction = self.model.predict(instance.values.reshape(1, -1))[0]
        
        if target_score is None:
            # Try to improve by 10%
            target_score = current_prediction * 1.1
        
        counterfactuals = []
        
        # Get feature importance
        local_importance = self.get_local_importance(instance_idx)
        top_features = local_importance.head(10)['Feature'].tolist()
        
        # Generate counterfactuals by modifying top features
        for feature in top_features[:n_counterfactuals]:
            cf_instance = instance.copy()
            
            # Modify feature (increase if negative impact, decrease if positive)
            feature_idx = self.feature_names.index(feature)
            local_shap = self.get_local_importance(instance_idx)
            feature_shap = local_shap[local_shap['Feature'] == feature]['SHAP_Value'].values[0]
            
            if feature_shap < 0:
                # Negative impact, try increasing
                cf_instance[feature] = instance[feature] * 1.2
            else:
                # Positive impact, try decreasing (to see what happens)
                cf_instance[feature] = instance[feature] * 0.8
            
            cf_prediction = self.model.predict(cf_instance.values.reshape(1, -1))[0]
            
            counterfactuals.append({
                'feature': feature,
                'original_value': instance[feature],
                'modified_value': cf_instance[feature],
                'change': cf_instance[feature] - instance[feature],
                'original_prediction': current_prediction,
                'counterfactual_prediction': cf_prediction,
                'prediction_change': cf_prediction - current_prediction,
                'target_achieved': cf_prediction >= target_score
            })
        
        counterfactuals_df = pd.DataFrame(counterfactuals)
        counterfactuals_df.to_csv(f'counterfactuals_instance_{instance_idx}.csv', index=False)
        
        print(f"Counterfactuals saved to 'counterfactuals_instance_{instance_idx}.csv'")
        return counterfactuals_df
    
    def prioritize_risk_factors(self, instance_idx, threshold_percentile=75):
        """Prioritize risk factors for a student."""
        print(f"\nPrioritizing risk factors for instance {instance_idx}...")
        
        local_importance = self.get_local_importance(instance_idx)
        current_prediction = self.model.predict(
            self.X_test.iloc[instance_idx:instance_idx+1]
        )[0]
        
        # Features with negative impact (decreasing score)
        risk_factors = local_importance[local_importance['SHAP_Value'] < 0].copy()
        risk_factors['Risk_Score'] = np.abs(risk_factors['SHAP_Value'])
        risk_factors = risk_factors.sort_values('Risk_Score', ascending=False)
        
        # Calculate threshold
        threshold = risk_factors['Risk_Score'].quantile(threshold_percentile / 100)
        high_risk_factors = risk_factors[risk_factors['Risk_Score'] >= threshold]
        
        risk_prioritization = {
            'student_id': instance_idx,
            'current_predicted_score': current_prediction,
            'high_risk_factors': high_risk_factors[['Feature', 'SHAP_Value', 'Risk_Score']].to_dict('records'),
            'all_risk_factors': risk_factors[['Feature', 'SHAP_Value', 'Risk_Score']].to_dict('records'),
            'recommendations': []
        }
        
        # Generate recommendations
        for _, factor in high_risk_factors.head(5).iterrows():
            feature_name = factor['Feature']
            recommendation = f"Focus on improving {feature_name} to increase predicted score"
            risk_prioritization['recommendations'].append(recommendation)
        
        return risk_prioritization
    
    def create_summary_plot(self, save_path='shap_summary.html'):
        """Create SHAP summary plot."""
        print("\nCreating SHAP summary plot...")
        
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Use plotly for interactive visualization
        if isinstance(self.shap_values, list):
            shap_values_array = np.array(self.shap_values)
        else:
            shap_values_array = self.shap_values
        
        # Get top features by mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:20]
        
        fig = go.Figure()
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_vals = shap_values_array[:, idx]
            
            fig.add_trace(go.Box(
                y=shap_vals,
                name=feature_name,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title="SHAP Summary Plot (Top 20 Features)",
            xaxis_title="Feature",
            yaxis_title="SHAP Value",
            height=600
        )
        
        fig.write_html(save_path)
        print(f"Summary plot saved to '{save_path}'")
        return fig


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    
    # Load data
    try:
        df = pd.read_csv('engineered_features_data.csv')
    except:
        try:
            df = pd.read_csv('preprocessed_data.csv')
        except:
            df = pd.read_csv('StudentPerformanceFactors.csv')
    
    # Prepare data
    target = 'Exam_Score'
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
    y = df[target].fillna(df[target].mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a model
    print("Training model for explainability...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    
    # Initialize explainable AI
    explainer = ExplainableAI(model, X_train, X_test, feature_names=X.columns.tolist())
    
    # Create SHAP explainer
    explainer.create_shap_explainer(explainer_type='tree')
    
    # Calculate SHAP values
    explainer.calculate_shap_values()
    
    # Global importance
    global_importance = explainer.get_global_importance()
    explainer.visualize_global_importance()
    
    # Local importance for first test instance
    local_importance = explainer.get_local_importance(0)
    explainer.visualize_local_importance(0)
    
    # Waterfall plot
    explainer.create_waterfall_plot(0)
    
    # Counterfactuals
    counterfactuals = explainer.generate_counterfactuals(0)
    
    # Risk factor prioritization
    risk_factors = explainer.prioritize_risk_factors(0)
    
    print("\nRisk Factor Prioritization:")
    print(f"Current Predicted Score: {risk_factors['current_predicted_score']:.2f}")
    print("\nHigh Risk Factors:")
    for factor in risk_factors['high_risk_factors'][:5]:
        print(f"  {factor['Feature']}: Risk Score = {factor['Risk_Score']:.4f}")
    
    print("\nRecommendations:")
    for rec in risk_factors['recommendations']:
        print(f"  - {rec}")
    
    # Summary plot
    explainer.create_summary_plot()
    
    print("\n" + "="*50)
    print("Module 7: Explainable AI & Counterfactuals Complete!")
    print("="*50)
