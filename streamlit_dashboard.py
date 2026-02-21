"""
Streamlit Dashboard for Student Performance Forecasting
Interactive UI with visualizations, drift monitoring, and explanations.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime
from collections import deque

# Page config
st.set_page_config(
    page_title="Student Performance Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

@st.cache_data
def load_data():
    """Load dataset."""
    try:
        df = pd.read_csv('StudentPerformanceFactors.csv')
        return df
    except:
        st.error("Could not load StudentPerformanceFactors.csv")
        return None

@st.cache_data
def load_preprocessed_data():
    """Load preprocessed data if available."""
    try:
        return pd.read_csv('preprocessed_data.csv')
    except:
        return None

@st.cache_data
def load_engineered_data():
    """Load engineered features data if available."""
    try:
        return pd.read_csv('engineered_features_data.csv')
    except:
        return None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">📊 Student Performance Forecasting Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📈 Data Overview", "🔍 EDA", "🤖 Model Performance", 
         "🎯 Ensemble Details", "🔄 Concept Drift", "💡 Explainability", "📊 Predictions"]
    )
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Please ensure StudentPerformanceFactors.csv is in the project directory.")
        return
    
    # Route to different pages
    if page == "🏠 Home":
        show_home(df)
    elif page == "📈 Data Overview":
        show_data_overview(df)
    elif page == "🔍 EDA":
        show_eda(df)
    elif page == "🤖 Model Performance":
        show_model_performance()
    elif page == "🎯 Ensemble Details":
        show_ensemble_details()
    elif page == "🔄 Concept Drift":
        show_concept_drift()
    elif page == "💡 Explainability":
        show_explainability()
    elif page == "📊 Predictions":
        show_predictions(df)

def show_home(df):
    """Home page."""
    st.header("Welcome to Student Performance Forecasting System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", len(df))
    
    with col2:
        st.metric("Features", len(df.columns) - 1)
    
    with col3:
        avg_score = df['Exam_Score'].mean()
        st.metric("Average Exam Score", f"{avg_score:.2f}")
    
    st.markdown("---")
    
    st.subheader("System Overview")
    st.markdown("""
    This system provides:
    - **Stacking Ensemble** for student performance prediction
    - **Base Models**: CatBoost, LightGBM, RandomForest, MLP
    - **Meta-Model**: Ridge/XGBoost/Neural Network learns from base predictions
    - **Concept Drift Detection** using ADWIN and Page-Hinkley tests
    - **Explainable AI** with SHAP values and counterfactual analysis
    - **Personalized Insights** for individual students
    """)
    
    st.subheader("Quick Stats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Distribution**")
        fig = px.histogram(df, x='Exam_Score', nbins=30, 
                          title="Distribution of Exam Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Top Correlated Features**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Exam_Score' in numeric_cols:
            numeric_cols.remove('Exam_Score')
        
        correlations = {}
        for col in numeric_cols[:10]:
            corr = abs(df[col].corr(df['Exam_Score']))
            correlations[col] = corr
        
        corr_df = pd.DataFrame({
            'Feature': list(correlations.keys()),
            'Correlation': list(correlations.values())
        }).sort_values('Correlation', ascending=False)
        
        fig = px.bar(corr_df, x='Correlation', y='Feature', 
                     orientation='h', title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)

def show_data_overview(df):
    """Data overview page."""
    st.header("Data Overview")
    
    tab1, tab2, tab3 = st.tabs(["Dataset", "Statistics", "Missing Values"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        with col2:
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with tab2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Categorical Features")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            st.write(f"**{col}:**")
            st.write(df[col].value_counts())
    
    with tab3:
        st.subheader("Missing Values Analysis")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        }).sort_values('Missing Count', ascending=False)
        
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        if missing.sum() > 0:
            fig = px.bar(missing_df[missing_df['Missing Count'] > 0], 
                        x='Column', y='Missing Count',
                        title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)

def show_eda(df):
    """EDA page."""
    st.header("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Distributions", "Relationships", "Performance Categories"])
    
    with tab1:
        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        fig = px.histogram(df, x=selected_feature, nbins=30, 
                          title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Relationships")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Exam_Score' in numeric_cols:
            numeric_cols.remove('Exam_Score')
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X Feature", numeric_cols)
        with col2:
            y_feature = st.selectbox("Y Feature", ['Exam_Score'] + numeric_cols)
        
        fig = px.scatter(df, x=x_feature, y=y_feature, 
                        color='Exam_Score' if y_feature != 'Exam_Score' else None,
                        title=f"{x_feature} vs {y_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Categories")
        
        # Create performance categories
        low_threshold = df['Exam_Score'].quantile(0.33)
        high_threshold = df['Exam_Score'].quantile(0.67)
        
        def categorize(score):
            if score < low_threshold:
                return 'Low Performance'
            elif score < high_threshold:
                return 'Medium Performance'
            else:
                return 'High Performance'
        
        df['Performance_Category'] = df['Exam_Score'].apply(categorize)
        
        category_counts = df['Performance_Category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                    title="Performance Category Distribution")
        st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def load_ensemble():
    """Load and initialize ensemble model."""
    try:
        from module5_adaptive_ensemble import AdaptiveMetaEnsemble
        import json
        
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
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and fit stacking ensemble
        ensemble = AdaptiveMetaEnsemble(
            meta_model_type='ridge',  # Options: 'ridge', 'xgboost', 'neural_network'
            n_folds=5,
            window_size=50,
            learning_rate=0.1
        )
        ensemble.fit(X_train, y_train)
        
        # Load saved state if available
        if os.path.exists('ensemble_state.json'):
            with open('ensemble_state.json', 'r') as f:
                state = json.load(f)
                ensemble.weights = state.get('weights', ensemble.weights)
                ensemble.confidence_scores = state.get('confidence_scores', ensemble.confidence_scores)
                # Restore performance history
                for name, history in state.get('performance_history', {}).items():
                    if name in ensemble.performance_history:
                        ensemble.performance_history[name] = deque(history, maxlen=ensemble.window_size)
        
        # Get predictions and metrics
        ensemble_pred, individual_preds = ensemble.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Convert y_test to numpy array for consistent indexing
        y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        # Calculate ensemble metrics
        ensemble_metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test_array, ensemble_pred)),
            'MAE': mean_absolute_error(y_test_array, ensemble_pred),
            'R²': r2_score(y_test_array, ensemble_pred),
            'predictions': ensemble_pred,
            'y_test': y_test_array
        }
        
        # Calculate individual model metrics
        individual_metrics = {}
        for name, pred in individual_preds.items():
            individual_metrics[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test_array, pred)),
                'MAE': mean_absolute_error(y_test_array, pred),
                'R²': r2_score(y_test_array, pred),
                'predictions': pred
            }
        
        # Update weights and confidence scores based on test performance
        # This enables display of adaptive concepts
        ensemble.update_weights(y_test_array, individual_preds)
        
        # Track ensemble error
        ensemble_error = np.sqrt(mean_squared_error(y_test_array, ensemble_pred))
        if 'MetaModel' in ensemble.performance_history:
            ensemble.performance_history['MetaModel'].append(ensemble_error)
        
        # Calculate confidence scores for all models
        for name in ensemble.base_models.keys():
            ensemble.calculate_confidence_score(name)
        if 'MetaModel' in ensemble.confidence_scores:
            # Calculate confidence for meta-model if we have history
            if len(ensemble.performance_history.get('MetaModel', [])) > 1:
                ensemble.calculate_confidence_score('MetaModel')
        
        return ensemble, ensemble_metrics, individual_metrics, X_test, y_test_array
    except Exception as e:
        st.error(f"Error loading ensemble: {str(e)}")
        return None, None, None, None, None

def show_ensemble_details():
    """Comprehensive ensemble details page."""
    st.header("🎯 Stacking Ensemble Details")
    
    # Load ensemble
    with st.spinner("Loading ensemble model and computing metrics..."):
        ensemble, ensemble_metrics, individual_metrics, X_test, y_test = load_ensemble()
    
    if ensemble is None or ensemble_metrics is None:
        st.error("Could not load ensemble. Please run module5_adaptive_ensemble.py first.")
        if st.button("Run Ensemble Module"):
            st.info("Please run: python module5_adaptive_ensemble.py")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Comparison", 
        "🤖 Meta-Model Outputs",
        "📈 Performance History",
        "🎯 Prediction Analysis"
    ])
    
    with tab1:
        st.subheader("Ensemble vs Individual Models Performance")
        
        # Create comparison dataframe
        comparison_data = []
        for name, metrics in individual_metrics.items():
            comparison_data.append({
                'Model': name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R²']
            })
        
        comparison_data.append({
            'Model': 'STACKING',
            'RMSE': ensemble_metrics['RMSE'],
            'MAE': ensemble_metrics['MAE'],
            'R²': ensemble_metrics['R²']
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display metrics table
        st.dataframe(comparison_df.style.highlight_max(subset=['R²'], color='lightgreen')
                    .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                    .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'R²': '{:.4f}'}),
                    use_container_width=True)
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ensemble RMSE", f"{ensemble_metrics['RMSE']:.4f}")
        with col2:
            st.metric("Ensemble MAE", f"{ensemble_metrics['MAE']:.4f}")
        with col3:
            st.metric("Ensemble R²", f"{ensemble_metrics['R²']:.4f}")
        with col4:
            best_individual_r2 = max([m['R²'] for m in individual_metrics.values()])
            improvement = ((ensemble_metrics['R²'] - best_individual_r2) / abs(best_individual_r2)) * 100
            st.metric("R² Improvement", f"{improvement:+.2f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, (name, metrics) in enumerate(individual_metrics.items()):
                fig.add_trace(go.Bar(
                    name=name,
                    x=['RMSE', 'MAE'],
                    y=[metrics['RMSE'], metrics['MAE']],
                    marker_color=colors[i % len(colors)]
                ))
            fig.add_trace(go.Bar(
                name='STACKING',
                x=['RMSE', 'MAE'],
                y=[ensemble_metrics['RMSE'], ensemble_metrics['MAE']],
                marker_color='#9467bd',
                marker_line=dict(color='black', width=2)
            ))
            fig.update_layout(
                title="RMSE and MAE Comparison",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            models = list(individual_metrics.keys()) + ['STACKING']
            r2_scores = [individual_metrics[m]['R²'] for m in individual_metrics.keys()] + [ensemble_metrics['R²']]
            colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
            fig.add_trace(go.Bar(
                x=models,
                y=r2_scores,
                marker_color=colors_bar,
                text=[f"{s:.4f}" for s in r2_scores],
                textposition='outside'
            ))
            fig.update_layout(
                title="R² Score Comparison",
                yaxis_title="R² Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Improvement metrics
        st.subheader("Improvement Over Individual Models")
        
        best_individual_rmse = min([m['RMSE'] for m in individual_metrics.values()])
        best_individual_mae = min([m['MAE'] for m in individual_metrics.values()])
        best_individual_r2 = max([m['R²'] for m in individual_metrics.values()])
        
        rmse_improvement = ((best_individual_rmse - ensemble_metrics['RMSE']) / best_individual_rmse) * 100
        mae_improvement = ((best_individual_mae - ensemble_metrics['MAE']) / best_individual_mae) * 100
        r2_improvement = ((ensemble_metrics['R²'] - best_individual_r2) / abs(best_individual_r2)) * 100 if best_individual_r2 != 0 else 0
        
        improvement_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²'],
            'Best Individual': [best_individual_rmse, best_individual_mae, best_individual_r2],
            'Ensemble': [ensemble_metrics['RMSE'], ensemble_metrics['MAE'], ensemble_metrics['R²']],
            'Improvement %': [rmse_improvement, mae_improvement, r2_improvement]
        })
        
        st.dataframe(improvement_df.style.format({
            'Best Individual': '{:.4f}',
            'Ensemble': '{:.4f}',
            'Improvement %': '{:.2f}%'
        }), use_container_width=True)
    
    with tab2:
        st.subheader("🤖 Meta-Model Outputs")
        
        # Show meta-model coefficients/feature importance
        st.markdown("### Meta-Model Feature Importance (Base Model Contributions)")
        
        # Get meta-model coefficients
        if hasattr(ensemble.meta_model, 'coef_'):
            # Ridge regression coefficients
            coefs = ensemble.meta_model.coef_
            intercept = ensemble.meta_model.intercept_ if hasattr(ensemble.meta_model, 'intercept_') else 0
            
            coef_df = pd.DataFrame({
                'Base Model': list(ensemble.base_models.keys()),
                'Coefficient': coefs,
                'Absolute Coefficient': np.abs(coefs),
                'Contribution %': (np.abs(coefs) / np.sum(np.abs(coefs)) * 100)
            }).sort_values('Absolute Coefficient', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(coef_df.style.format({
                    'Coefficient': '{:.4f}',
                    'Absolute Coefficient': '{:.4f}',
                    'Contribution %': '{:.2f}%'
            }), use_container_width=True)
            
            with col2:
                # Visualization of coefficients
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=coef_df['Base Model'],
                    y=coef_df['Coefficient'],
                    marker_color=['green' if c > 0 else 'red' for c in coef_df['Coefficient']],
                    text=[f"{c:.4f}" for c in coef_df['Coefficient']],
                    textposition='outside'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title="Meta-Model Coefficients (How Base Models are Combined)",
                    xaxis_title="Base Model",
                    yaxis_title="Coefficient Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Intercept:** {intercept:.4f} | **Formula:** Final = {intercept:.4f} + " + 
                   " + ".join([f"({coef:.4f} × {name})" for name, coef in zip(coef_df['Base Model'], coef_df['Coefficient'])]))
        
        elif hasattr(ensemble.meta_model, 'feature_importances_'):
            # XGBoost feature importance
            importances = ensemble.meta_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Base Model': list(ensemble.base_models.keys()),
                'Importance': importances,
                'Importance %': (importances / np.sum(importances) * 100)
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(importance_df.style.format({
                    'Importance': '{:.4f}',
                    'Importance %': '{:.2f}%'
                }), use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['Base Model'],
                    y=importance_df['Importance'],
                    marker_color='skyblue',
                    text=[f"{i:.2f}%" for i in importance_df['Importance %']],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Meta-Model Feature Importance (Base Model Contributions)",
                    xaxis_title="Base Model",
                    yaxis_title="Importance",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
        else:
            st.info("Meta-model feature importance not available for this model type.")
        
        st.markdown("---")
        
        # Show sample predictions flow
        st.markdown("### Prediction Flow Example")
        
        if y_test is not None and len(y_test) > 0:
            # Get a sample prediction
            sample_idx = 0
            sample_size = min(10, len(y_test))
            
            # Get base model predictions for sample
            sample_base_preds = {}
            for name, metrics in individual_metrics.items():
                sample_base_preds[name] = metrics['predictions'][:sample_size]
            
            sample_meta_pred = ensemble_metrics['predictions'][:sample_size]
            sample_actual = ensemble_metrics['y_test'][:sample_size]
            
            # Create prediction flow dataframe
            flow_data = []
            for i in range(sample_size):
                row = {
                    'Sample': i + 1,
                    'Actual': sample_actual[i]
                }
                for name in ensemble.base_models.keys():
                    row[f'{name} Pred'] = sample_base_preds[name][i]
                row['Meta-Model (Final)'] = sample_meta_pred[i]
                row['Error'] = abs(sample_actual[i] - sample_meta_pred[i])
                flow_data.append(row)
            
            flow_df = pd.DataFrame(flow_data)
            
            st.dataframe(
                flow_df.style.format({
                    'Actual': '{:.2f}',
                    'CatBoost Pred': '{:.2f}',
                    'LightGBM Pred': '{:.2f}',
                    'RandomForest Pred': '{:.2f}',
                    'MLP Pred': '{:.2f}',
                    'Meta-Model (Final)': '{:.2f}',
                    'Error': '{:.2f}'
                }).background_gradient(subset=['Error'], cmap='YlOrRd'),
                use_container_width=True
            )
            
            # Visualization of prediction flow
            st.markdown("#### Prediction Flow Visualization (First 5 Samples)")
            fig = go.Figure()
            
            sample_indices = list(range(min(5, sample_size)))
            x_pos = np.arange(len(sample_indices))
            width = 0.15
            
            # Actual values
            fig.add_trace(go.Bar(
                x=x_pos,
                y=[sample_actual[i] for i in sample_indices],
                name='Actual',
                marker_color='black',
                width=width
            ))
            
            # Base model predictions
            colors_base = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for idx, name in enumerate(ensemble.base_models.keys()):
                fig.add_trace(go.Bar(
                    x=x_pos + width * (idx + 1),
                    y=[sample_base_preds[name][i] for i in sample_indices],
                    name=name,
                    marker_color=colors_base[idx % len(colors_base)],
                    width=width
                ))
            
            # Meta-model prediction
            fig.add_trace(go.Bar(
                x=x_pos + width * (len(ensemble.base_models) + 1),
                y=[sample_meta_pred[i] for i in sample_indices],
                name='Meta-Model (Final)',
                marker_color='#9467bd',
                marker_line=dict(color='black', width=2),
                width=width
            ))
            
            fig.update_layout(
                title="Prediction Flow: Base Models → Meta-Model → Final Prediction",
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_pos + width * (len(ensemble.base_models) + 1) / 2,
                    ticktext=[f'Sample {i+1}' for i in sample_indices]
                ),
                yaxis_title="Predicted Value",
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Performance History")
        
        # Load performance history
        if os.path.exists('ensemble_state.json'):
            import json
            with open('ensemble_state.json', 'r') as f:
                state = json.load(f)
                performance_history = state.get('performance_history', {})
        else:
            performance_history = {}
        
        if performance_history:
            # Create performance history dataframe
            max_len = max([len(history) for history in performance_history.values()])
            history_data = {}
            
            for model_name, history in performance_history.items():
                history_data[model_name] = history + [None] * (max_len - len(history))
            
            history_df = pd.DataFrame(history_data)
            
            st.write("**Performance History (RMSE over time)**")
            
            # Line chart
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, (model_name, history) in enumerate(performance_history.items()):
                fig.add_trace(go.Scatter(
                    x=list(range(len(history))),
                    y=history,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            fig.update_layout(
                title="Model Performance History (RMSE)",
                xaxis_title="Update Iteration",
                yaxis_title="RMSE",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("Performance Statistics")
            stats_data = []
            for model_name, history in performance_history.items():
                if history:
                    stats_data.append({
                        'Model': model_name,
                        'Mean RMSE': np.mean(history),
                        'Std RMSE': np.std(history),
                        'Min RMSE': np.min(history),
                        'Max RMSE': np.max(history),
                        'Latest RMSE': history[-1]
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.style.format({
                    'Mean RMSE': '{:.4f}',
                    'Std RMSE': '{:.4f}',
                    'Min RMSE': '{:.4f}',
                    'Max RMSE': '{:.4f}',
                    'Latest RMSE': '{:.4f}'
                }), use_container_width=True)
        else:
            st.info("No performance history available. Performance history is generated during ensemble training and weight updates.")
    
    with tab4:
        st.subheader("Prediction Analysis")
        
        if y_test is not None and ensemble_metrics is not None:
            # Prediction vs Actual scatter
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Stacking Ensemble Predictions vs Actual**")
                # Use y_test from ensemble_metrics for consistency
                y_test_vals = ensemble_metrics['y_test']
                pred_vals = ensemble_metrics['predictions']
                sample_limit = min(100, len(y_test_vals))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test_vals[:sample_limit],  # Show first 100 for clarity
                    y=pred_vals[:sample_limit],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=5)
                ))
                # Perfect prediction line
                min_val = min(min(y_test_vals[:sample_limit]), min(pred_vals[:sample_limit]))
                max_val = max(max(y_test_vals[:sample_limit]), max(pred_vals[:sample_limit]))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Stacking: Predicted vs Actual",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Residuals Distribution**")
                y_test_vals = ensemble_metrics['y_test']
                pred_vals = ensemble_metrics['predictions']
                sample_limit = min(100, len(y_test_vals))
                residuals = y_test_vals[:sample_limit] - pred_vals[:sample_limit]
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker_color='skyblue'
                ))
                fig.update_layout(
                    title="Residuals Distribution",
                    xaxis_title="Residuals",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual model predictions comparison
            st.write("**Individual Model Predictions Comparison**")
            y_test_vals = ensemble_metrics['y_test']
            sample_size = min(50, len(y_test_vals))
            sample_indices = np.random.choice(len(y_test_vals), sample_size, replace=False)
            
            # Convert range to numpy array for Plotly (more reliable than list)
            x_indices = np.arange(sample_size)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=y_test_vals[sample_indices],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=2),
                marker=dict(size=8)
            ))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, (name, metrics) in enumerate(individual_metrics.items()):
                fig.add_trace(go.Scatter(
                    x=x_indices,
                    y=metrics['predictions'][sample_indices],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    marker=dict(size=5)
                ))
            
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=ensemble_metrics['predictions'][sample_indices],
                mode='lines+markers',
                name='STACKING',
                line=dict(color='#9467bd', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Predictions Comparison (Sample)",
                xaxis_title="Sample Index",
                yaxis_title="Predicted/Actual Value",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Prediction analysis requires test data. Please ensure the ensemble has been trained.")

def show_model_performance():
    """Model performance page."""
    st.header("Model Performance")
    
    try:
        comparison_df = pd.read_csv('baseline_model_comparison.csv')
        
        st.subheader("Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Model', y='Test_RMSE',
                        title="Test RMSE by Model")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='Test_R²',
                        title="Test R² by Model")
            st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Model comparison results not found. Please run the baseline models module first.")
        if st.button("Run Model Training"):
            st.info("Please run: python module4_baseline_models.py")

def show_concept_drift():
    """Concept drift monitoring page."""
    st.header("Concept Drift Detection")
    
    st.info("Concept drift detection monitors changes in data distribution over time.")
    
    st.subheader("Drift Detection Status")
    
    summary_path = "concept_drift_summary.csv"
    history_path = "concept_drift_history.csv"
    
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            summary = summary_df.iloc[0].to_dict()
        else:
            summary = {
                "adwin_detections": 0,
                "ph_detections": 0,
                "total_drifts": 0,
            }
    else:
        summary = {
            "adwin_detections": 0,
            "ph_detections": 0,
            "total_drifts": 0,
        }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ADWIN Detections", int(summary.get("adwin_detections", 0)))
    with col2:
        st.metric("Page-Hinkley Detections", int(summary.get("ph_detections", 0)))
    with col3:
        st.metric("Total Drifts", int(summary.get("total_drifts", 0)))
    
    st.subheader("Drift History")
    
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        if history_df.empty:
            st.info("No drift events detected in the monitored stream.")
        else:
            st.dataframe(history_df, use_container_width=True)
            
            # Simple visualization of drift types
            if "drift_type" in history_df.columns:
                type_counts = history_df["drift_type"].value_counts().reset_index()
                type_counts.columns = ["Drift Type", "Count"]
                fig = px.bar(type_counts, x="Drift Type", y="Count",
                             title="Drift Types Detected")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Run the concept drift detection module "
            "(`python module6_concept_drift.py` or `python main.py`) "
            "to generate drift history."
        )

def show_explainability():
    """Explainability page."""
    st.header("Explainable AI & Counterfactuals")
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Values", "Counterfactuals"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        try:
            importance_df = pd.read_csv('global_feature_importance.csv')
            fig = px.bar(importance_df.head(20), x='Importance', y='Feature',
                        orientation='h', title="Top 20 Features by Importance")
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("Feature importance not found. Run the explainability module first.")
    
    with tab2:
        st.subheader("SHAP Analysis")
        st.info("SHAP values show the contribution of each feature to predictions.")
        
        shap_files = {
            "Summary Plot": "shap_summary.html",
            "Global Importance (HTML)": "global_importance.html",
            "Local Importance (HTML)": "local_importance.html",
            "Waterfall Plot": "waterfall_plot.html",
        }
        
        available_files = {
            name: path for name, path in shap_files.items() if os.path.exists(path)
        }
        
        if not available_files:
            st.info("Run `python module7_explainable_ai.py` (or the full pipeline via `python main.py`) to generate SHAP visualizations.")
        else:
            selected = st.selectbox("Select SHAP visualization", list(available_files.keys()))
            html_path = available_files[selected]
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=700, scrolling=True)
            except Exception as e:
                st.error(f"Could not load SHAP visualization from {html_path}: {e}")
    
    with tab3:
        st.subheader("Counterfactual Scenarios")
        st.info("Counterfactuals show what changes would improve a student's predicted score.")
        
        # Support multiple counterfactual files if available (per instance)
        cf_files = [f for f in os.listdir(".") if f.startswith("counterfactuals_instance_") and f.endswith(".csv")]
        
        if not cf_files:
            st.info("Run `python module7_explainable_ai.py` (or the full pipeline via `python main.py`) to generate counterfactuals.")
        else:
            # Map files to instance indices for nicer labels
            options = {}
            for fname in cf_files:
                try:
                    # Expect pattern: counterfactuals_instance_{idx}.csv
                    idx_part = fname.replace("counterfactuals_instance_", "").replace(".csv", "")
                    idx = int(idx_part)
                    label = f"Instance {idx}"
                except Exception:
                    label = fname
                options[label] = fname
            
            selected_label = st.selectbox("Select counterfactual instance", list(options.keys()))
            selected_file = options[selected_label]
            
            try:
                cf_df = pd.read_csv(selected_file)
                st.dataframe(cf_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load counterfactuals from {selected_file}: {e}")

def show_predictions(df):
    """Predictions page."""
    st.header("Make Predictions")
    
    st.subheader("Input Student Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hours_studied = st.slider("Hours Studied", 0, 50, 20)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        previous_scores = st.slider("Previous Scores", 0, 100, 75)
    
    with col2:
        parental_involvement = st.selectbox("Parental Involvement", 
                                           ["Low", "Medium", "High"])
        access_to_resources = st.selectbox("Access to Resources", 
                                          ["Low", "Medium", "High"])
        motivation_level = st.selectbox("Motivation Level", 
                                       ["Low", "Medium", "High"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    
    with col3:
        tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 2)
        family_income = st.selectbox("Family Income", 
                                   ["Low", "Medium", "High"])
        teacher_quality = st.selectbox("Teacher Quality", 
                                      ["Low", "Medium", "High"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
    
    if st.button("Predict Exam Score"):
        st.info("Model prediction functionality requires trained models.")
        st.info("Please run the model training modules first.")
        
        # Placeholder prediction
        st.success(f"Predicted Exam Score: {previous_scores:.1f} (placeholder)")

if __name__ == "__main__":
    main()
