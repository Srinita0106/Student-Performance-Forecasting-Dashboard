"""
Module 4: Baseline Model Comparison
Compares XGBoost, LightGBM, and CatBoost models with RMSE, MAE, R² metrics
and cross-validation stability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

class BaselineModelComparison:
    def __init__(self, df, target='Exam_Score', test_size=0.2, random_state=42):
        """Initialize model comparison."""
        self.df = df.copy()
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        """Prepare train-test split."""
        print("\nPreparing data...")
        
        # Separate features and target
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe")
        
        X = self.df.drop(columns=[self.target])
        
        # Remove non-numeric columns for baseline models
        X = X.select_dtypes(include=[np.number])
        
        y = self.df[self.target]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost(self):
        """Train XGBoost model."""
        print("\nTraining XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.results['XGBoost'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }
        
        print(f"XGBoost - Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
        
        return model
    
    def train_lightgbm(self):
        """Train LightGBM model."""
        print("\nTraining LightGBM...")
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['LightGBM'] = model
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.results['LightGBM'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }
        
        print(f"LightGBM - Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
        
        return model
    
    def train_catboost(self):
        """Train CatBoost model."""
        print("\nTraining CatBoost...")
        
        model = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=False
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['CatBoost'] = model
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.results['CatBoost'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }
        
        print(f"CatBoost - Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
        
        return model
    
    def cross_validate_models(self, cv_folds=5):
        """Perform cross-validation for all models."""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nCross-validating {model_name}...")
            
            # RMSE scores (negative because sklearn maximizes)
            rmse_scores = np.sqrt(-cross_val_score(
                model, self.X_train, self.y_train,
                cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
            ))
            
            # R² scores
            r2_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=kfold, scoring='r2', n_jobs=-1
            )
            
            cv_results[model_name] = {
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'rmse_scores': rmse_scores,
                'r2_scores': r2_scores
            }
            
            print(f"{model_name} CV - RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
            print(f"{model_name} CV - R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        
        self.results['cv_results'] = cv_results
        return cv_results
    
    def compare_models(self):
        """Compare all models and create summary."""
        print("\n" + "="*50)
        print("Model Comparison Summary")
        print("="*50)
        
        comparison_data = []
        
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            if model_name in self.results:
                result = self.results[model_name]
                comparison_data.append({
                    'Model': model_name,
                    'Test_RMSE': result['test_rmse'],
                    'Test_MAE': result['test_mae'],
                    'Test_R²': result['test_r2'],
                    'Train_RMSE': result['train_rmse'],
                    'Train_MAE': result['train_mae'],
                    'Train_R²': result['train_r2']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_RMSE')
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Add CV results if available
        if 'cv_results' in self.results:
            cv_summary = []
            for model_name, cv_result in self.results['cv_results'].items():
                cv_summary.append({
                    'Model': model_name,
                    'CV_RMSE_Mean': cv_result['rmse_mean'],
                    'CV_RMSE_Std': cv_result['rmse_std'],
                    'CV_R²_Mean': cv_result['r2_mean'],
                    'CV_R²_Std': cv_result['r2_std']
                })
            
            cv_df = pd.DataFrame(cv_summary)
            print("\nCross-Validation Results:")
            print(cv_df.to_string(index=False))
            
            comparison_df = comparison_df.merge(cv_df, on='Model', how='left')
        
        comparison_df.to_csv('baseline_model_comparison.csv', index=False)
        print("\nComparison saved to 'baseline_model_comparison.csv'")
        
        return comparison_df
    
    def get_best_model(self):
        """Get the best performing model based on test RMSE."""
        if not self.results:
            return None
        
        best_model_name = min(
            [name for name in self.results.keys() if name != 'cv_results'],
            key=lambda x: self.results[x]['test_rmse']
        )
        
        return self.models[best_model_name], best_model_name
    
    def train_all_models(self):
        """Train all baseline models."""
        print("\n" + "="*50)
        print("Training Baseline Models")
        print("="*50)
        
        # Prepare data
        self.prepare_data()
        
        # Train models
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        
        # Cross-validation
        self.cross_validate_models()
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Get best model
        best_model, best_name = self.get_best_model()
        print(f"\nBest Model: {best_name}")
        print(f"Best Test RMSE: {self.results[best_name]['test_rmse']:.4f}")
        
        return best_model, best_name


if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv('engineered_features_data.csv')
    except:
        try:
            df = pd.read_csv('data_with_performance_categories.csv')
        except:
            try:
                df = pd.read_csv('preprocessed_data.csv')
            except:
                df = pd.read_csv('StudentPerformanceFactors.csv')
    
    # Initialize model comparison
    model_comp = BaselineModelComparison(df, target='Exam_Score')
    
    # Train all models
    best_model, best_name = model_comp.train_all_models()
    
    print("\n" + "="*50)
    print("Module 4: Baseline Model Comparison Complete!")
    print("="*50)
