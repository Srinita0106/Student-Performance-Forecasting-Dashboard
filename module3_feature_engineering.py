"""
Module 3: Advanced Feature Engineering
Creates engineered features: Engagement Index, Behavioral Stability Score,
Support Score, Academic Momentum, Consistency Index, Weighted Engagement,
Student Risk Profile, and temporal features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, df):
        """Initialize feature engineer with dataframe."""
        self.df = df.copy()
        self.target = 'Exam_Score'
        self.engineered_features = []
        
    def create_engagement_index(self):
        """Create Engagement Index based on study hours, attendance, and motivation."""
        print("\nCreating Engagement Index...")
        
        # Normalize components
        scaler = MinMaxScaler()
        
        # Components of engagement
        hours_studied = self.df['Hours_Studied'].values.reshape(-1, 1)
        attendance = self.df['Attendance'].values.reshape(-1, 1)
        
        # Normalize
        hours_norm = scaler.fit_transform(hours_studied).flatten()
        attendance_norm = scaler.fit_transform(attendance).flatten()
        
        # Motivation encoding (if exists)
        if 'Motivation_Level' in self.df.columns:
            motivation_map = {'Low': 0.33, 'Medium': 0.67, 'High': 1.0}
            motivation_norm = self.df['Motivation_Level'].map(motivation_map).fillna(0.5)
        else:
            motivation_norm = pd.Series([0.5] * len(self.df))
        
        # Weighted combination
        self.df['Engagement_Index'] = (
            0.4 * hours_norm +
            0.4 * attendance_norm +
            0.2 * motivation_norm
        )
        
        self.engineered_features.append('Engagement_Index')
        print("Engagement Index created.")
        return self.df
    
    def create_behavioral_stability_score(self):
        """Create Behavioral Stability Score based on consistency in attendance and study patterns."""
        print("\nCreating Behavioral Stability Score...")
        
        # Calculate rolling standard deviation for attendance
        if 'Attendance' in self.df.columns:
            # For demonstration, use overall std relative to mean
            attendance_cv = self.df['Attendance'].std() / (self.df['Attendance'].mean() + 1e-6)
            attendance_stability = 1 / (1 + attendance_cv)
        else:
            attendance_stability = pd.Series([0.5] * len(self.df))
        
        # Study hours consistency
        if 'Hours_Studied' in self.df.columns:
            hours_cv = self.df['Hours_Studied'].std() / (self.df['Hours_Studied'].mean() + 1e-6)
            hours_stability = 1 / (1 + hours_cv)
        else:
            hours_stability = pd.Series([0.5] * len(self.df))
        
        # Combine stability scores
        self.df['Behavioral_Stability_Score'] = (
            0.5 * attendance_stability +
            0.5 * hours_stability
        )
        
        self.engineered_features.append('Behavioral_Stability_Score')
        print("Behavioral Stability Score created.")
        return self.df
    
    def create_support_score(self):
        """Create Support Score based on parental involvement, resources, and tutoring."""
        print("\nCreating Support Score...")
        
        support_components = []
        
        # Parental involvement
        if 'Parental_Involvement' in self.df.columns:
            parental_map = {'Low': 0.33, 'Medium': 0.67, 'High': 1.0}
            parental_score = self.df['Parental_Involvement'].map(parental_map).fillna(0.5)
            support_components.append(parental_score)
        
        # Access to resources
        if 'Access_to_Resources' in self.df.columns:
            resources_map = {'Low': 0.33, 'Medium': 0.67, 'High': 1.0}
            resources_score = self.df['Access_to_Resources'].map(resources_map).fillna(0.5)
            support_components.append(resources_score)
        
        # Tutoring sessions
        if 'Tutoring_Sessions' in self.df.columns:
            scaler = MinMaxScaler()
            tutoring_norm = scaler.fit_transform(
                self.df['Tutoring_Sessions'].values.reshape(-1, 1)
            ).flatten()
            support_components.append(pd.Series(tutoring_norm))
        
        # Family income
        if 'Family_Income' in self.df.columns:
            income_map = {'Low': 0.33, 'Medium': 0.67, 'High': 1.0}
            income_score = self.df['Family_Income'].map(income_map).fillna(0.5)
            support_components.append(income_score)
        
        if support_components:
            self.df['Support_Score'] = pd.concat(support_components, axis=1).mean(axis=1)
        else:
            self.df['Support_Score'] = 0.5
        
        self.engineered_features.append('Support_Score')
        print("Support Score created.")
        return self.df
    
    def create_academic_momentum(self):
        """Create Academic Momentum based on previous scores and current trajectory."""
        print("\nCreating Academic Momentum...")
        
        if 'Previous_Scores' in self.df.columns:
            # Calculate momentum as improvement potential
            # Higher previous scores with good current indicators = positive momentum
            prev_scores = self.df['Previous_Scores']
            
            # Normalize previous scores
            scaler = MinMaxScaler()
            prev_norm = scaler.fit_transform(prev_scores.values.reshape(-1, 1)).flatten()
            
            # Combine with engagement
            if 'Engagement_Index' in self.df.columns:
                momentum = (prev_norm * 0.6 + self.df['Engagement_Index'] * 0.4)
            else:
                momentum = prev_norm
            
            self.df['Academic_Momentum'] = momentum
        else:
            self.df['Academic_Momentum'] = 0.5
        
        self.engineered_features.append('Academic_Momentum')
        print("Academic Momentum created.")
        return self.df
    
    def create_consistency_index(self):
        """Create Consistency Index measuring regularity in academic behavior."""
        print("\nCreating Consistency Index...")
        
        consistency_components = []
        
        # Sleep consistency
        if 'Sleep_Hours' in self.df.columns:
            sleep_mean = self.df['Sleep_Hours'].mean()
            sleep_std = self.df['Sleep_Hours'].std()
            sleep_consistency = 1 / (1 + abs(self.df['Sleep_Hours'] - sleep_mean) / (sleep_std + 1e-6))
            consistency_components.append(sleep_consistency)
        
        # Study hours consistency
        if 'Hours_Studied' in self.df.columns:
            hours_mean = self.df['Hours_Studied'].mean()
            hours_std = self.df['Hours_Studied'].std()
            hours_consistency = 1 / (1 + abs(self.df['Hours_Studied'] - hours_mean) / (hours_std + 1e-6))
            consistency_components.append(hours_consistency)
        
        # Attendance consistency
        if 'Attendance' in self.df.columns:
            att_mean = self.df['Attendance'].mean()
            att_std = self.df['Attendance'].std()
            att_consistency = 1 / (1 + abs(self.df['Attendance'] - att_mean) / (att_std + 1e-6))
            consistency_components.append(att_consistency)
        
        if consistency_components:
            self.df['Consistency_Index'] = pd.concat(consistency_components, axis=1).mean(axis=1)
        else:
            self.df['Consistency_Index'] = 0.5
        
        self.engineered_features.append('Consistency_Index')
        print("Consistency Index created.")
        return self.df
    
    def create_weighted_engagement(self):
        """Create Weighted Engagement combining multiple engagement factors."""
        print("\nCreating Weighted Engagement...")
        
        weights = {}
        components = []
        
        if 'Engagement_Index' in self.df.columns:
            components.append(self.df['Engagement_Index'])
            weights[len(components)-1] = 0.3
        
        if 'Behavioral_Stability_Score' in self.df.columns:
            components.append(self.df['Behavioral_Stability_Score'])
            weights[len(components)-1] = 0.2
        
        if 'Consistency_Index' in self.df.columns:
            components.append(self.df['Consistency_Index'])
            weights[len(components)-1] = 0.2
        
        if 'Academic_Momentum' in self.df.columns:
            components.append(self.df['Academic_Momentum'])
            weights[len(components)-1] = 0.3
        
        if components:
            weighted_sum = sum(components[i] * weights.get(i, 1/len(components)) 
                             for i in range(len(components)))
            total_weight = sum(weights.values()) if weights else len(components)
            self.df['Weighted_Engagement'] = weighted_sum / total_weight
        else:
            self.df['Weighted_Engagement'] = 0.5
        
        self.engineered_features.append('Weighted_Engagement')
        print("Weighted Engagement created.")
        return self.df
    
    def create_student_risk_profile(self):
        """Create Student Risk Profile based on multiple risk factors."""
        print("\nCreating Student Risk Profile...")
        
        risk_factors = []
        
        # Low engagement risk
        if 'Engagement_Index' in self.df.columns:
            engagement_risk = 1 - self.df['Engagement_Index']
            risk_factors.append(engagement_risk)
        
        # Low support risk
        if 'Support_Score' in self.df.columns:
            support_risk = 1 - self.df['Support_Score']
            risk_factors.append(support_risk)
        
        # Low previous performance risk
        if 'Previous_Scores' in self.df.columns:
            scaler = MinMaxScaler()
            prev_norm = scaler.fit_transform(
                self.df['Previous_Scores'].values.reshape(-1, 1)
            ).flatten()
            performance_risk = pd.Series(1 - prev_norm, index=self.df.index)
            risk_factors.append(performance_risk)
        
        # Learning disabilities risk
        if 'Learning_Disabilities' in self.df.columns:
            disability_risk = (self.df['Learning_Disabilities'] == 'Yes').astype(int) * 0.5
            risk_factors.append(pd.Series(disability_risk, index=self.df.index))
        
        # Low attendance risk
        if 'Attendance' in self.df.columns:
            scaler = MinMaxScaler()
            att_norm = scaler.fit_transform(
                self.df['Attendance'].values.reshape(-1, 1)
            ).flatten()
            attendance_risk = pd.Series(1 - att_norm, index=self.df.index)
            risk_factors.append(attendance_risk)
        
        if risk_factors:
            self.df['Student_Risk_Profile'] = pd.concat(risk_factors, axis=1).mean(axis=1)
        else:
            self.df['Student_Risk_Profile'] = 0.5
        
        self.engineered_features.append('Student_Risk_Profile')
        print("Student Risk Profile created.")
        return self.df
    
    def create_temporal_features(self):
        """Create temporal features: rolling means, trend slopes, performance velocity."""
        print("\nCreating temporal features...")
        
        # Sort by index to simulate temporal order
        self.df = self.df.sort_index()
        
        # Rolling attendance mean (window of 5)
        if 'Attendance' in self.df.columns:
            self.df['Rolling_Attendance_Mean'] = self.df['Attendance'].rolling(
                window=min(5, len(self.df)), min_periods=1
            ).mean()
        
        # Trend slope for study hours
        if 'Hours_Studied' in self.df.columns:
            window = min(5, len(self.df))
            slopes = []
            for i in range(len(self.df)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                if end_idx - start_idx > 1:
                    x = np.arange(end_idx - start_idx)
                    y = self.df['Hours_Studied'].iloc[start_idx:end_idx].values
                    slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                else:
                    slope = 0
                slopes.append(slope)
            self.df['Study_Hours_Trend_Slope'] = slopes
        
        # Performance velocity (change in previous scores)
        if 'Previous_Scores' in self.df.columns:
            self.df['Performance_Velocity'] = self.df['Previous_Scores'].diff().fillna(0)
        
        temporal_features = ['Rolling_Attendance_Mean', 'Study_Hours_Trend_Slope', 
                            'Performance_Velocity']
        self.engineered_features.extend([f for f in temporal_features if f in self.df.columns])
        
        print("Temporal features created.")
        return self.df
    
    def analyze_feature_importance(self, model=None):
        """Analyze feature importance if model is provided."""
        print("\nAnalyzing feature importance...")
        
        if model is None:
            # Use correlation as proxy
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target in numerical_cols:
                numerical_cols.remove(self.target)
            
            importance = {}
            for col in numerical_cols:
                corr = abs(self.df[col].corr(self.df[self.target]))
                importance[col] = corr
            
            importance_df = pd.Series(importance).sort_values(ascending=False)
        else:
            # Use model feature importance
            if hasattr(model, 'feature_importances_'):
                feature_names = self.df.drop(columns=[self.target]).columns
                importance_df = pd.Series(
                    model.feature_importances_,
                    index=feature_names
                ).sort_values(ascending=False)
            else:
                print("Model does not have feature_importances_ attribute.")
                return None
        
        importance_df.to_csv('feature_importance.csv')
        print("Feature importance saved to 'feature_importance.csv'")
        return importance_df
    
    def get_engineered_features(self):
        """Return list of engineered features."""
        return self.engineered_features
    
    def save_engineered_data(self, file_path='engineered_features_data.csv'):
        """Save data with engineered features."""
        self.df.to_csv(file_path, index=False)
        print(f"\nEngineered features data saved to '{file_path}'")
        return self.df


if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv('data_with_performance_categories.csv')
    except:
        try:
            df = pd.read_csv('preprocessed_data.csv')
        except:
            df = pd.read_csv('StudentPerformanceFactors.csv')
    
    # Initialize feature engineer
    fe = FeatureEngineer(df)
    
    # Create all engineered features
    fe.create_engagement_index()
    fe.create_behavioral_stability_score()
    fe.create_support_score()
    fe.create_academic_momentum()
    fe.create_consistency_index()
    fe.create_weighted_engagement()
    fe.create_student_risk_profile()
    fe.create_temporal_features()
    
    # Analyze feature importance
    fe.analyze_feature_importance()
    
    # Save engineered data
    fe.save_engineered_data()
    
    print("\n" + "="*50)
    print("Module 3: Feature Engineering Complete!")
    print(f"Created {len(fe.get_engineered_features())} engineered features")
    print("="*50)
