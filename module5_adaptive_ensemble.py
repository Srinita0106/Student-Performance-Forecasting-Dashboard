"""
Module 5: Stacking Ensemble
Uses stacking with base models (CatBoost, LightGBM, RandomForest, MLP)
and a meta-model (Ridge/XGBoost/Neural Network) for improved performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AdaptiveMetaEnsemble:
    def __init__(self, base_models=None, meta_model_type='ridge', n_folds=5, 
                 window_size=100, learning_rate=0.1, random_state=42):
        """
        Initialize stacking ensemble.
        
        Parameters:
        -----------
        base_models : dict, optional
            Dictionary of base models. If None, uses default: CatBoost, LightGBM, RandomForest, MLP
        meta_model_type : str, default='ridge'
            Type of meta-model: 'ridge', 'xgboost', or 'neural_network'
        n_folds : int, default=5
            Number of folds for cross-validation in stacking
        window_size : int, default=100
            Window size for performance tracking
        learning_rate : float, default=0.1
            Learning rate (for adaptive features, if needed)
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_folds = n_folds
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.meta_model_type = meta_model_type
        
        # Initialize base models if not provided
        if base_models is None:
            self.base_models = {
                'CatBoost': cb.CatBoostRegressor(
                    iterations=100, depth=6, learning_rate=0.1,
                    random_state=random_state, verbose=False
                ),
                'LightGBM': lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=random_state, n_jobs=-1, verbosity=-1
                ),
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=10,
                    random_state=random_state, n_jobs=-1
                ),
                'MLP': MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500,
                    random_state=random_state, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=10
                )
            }
        else:
            self.base_models = base_models
        
        # Initialize meta-model
        self.meta_model = self._create_meta_model(meta_model_type)
        
        # Performance history for tracking
        self.performance_history = {name: deque(maxlen=window_size) 
                                   for name in self.base_models.keys()}
        self.performance_history['MetaModel'] = deque(maxlen=window_size)
        
        # Confidence scores (for compatibility)
        self.confidence_scores = {name: 1.0 for name in self.base_models.keys()}
        self.confidence_scores['MetaModel'] = 1.0
        
        # Weights (for compatibility, but not used in stacking)
        self.weights = {name: 1.0 / len(self.base_models) 
                       for name in self.base_models.keys()}
        
    def _create_meta_model(self, meta_model_type):
        """Create meta-model based on type."""
        if meta_model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=self.random_state)
        elif meta_model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1, verbosity=0
            )
        elif meta_model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=(50, 25), max_iter=300,
                random_state=self.random_state, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=10
            )
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}. Choose 'ridge', 'xgboost', or 'neural_network'")
    
    def fit(self, X_train, y_train):
        """
        Fit stacking ensemble using cross-validation.
        
        Steps:
        1. Train base models using cross-validation to get out-of-fold predictions
        2. Use out-of-fold predictions as features for meta-model
        3. Train meta-model on these features
        4. Retrain base models on full training set
        """
        print("\n" + "="*70)
        print("Fitting Stacking Ensemble")
        print("="*70)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Initialize KFold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Step 1: Generate out-of-fold predictions for meta-model training
        print(f"\nGenerating out-of-fold predictions using {self.n_folds}-fold CV...")
        meta_features = np.zeros((len(X_train), len(self.base_models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"  Processing fold {fold_idx + 1}/{self.n_folds}...")
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train each base model on fold training data
            for model_idx, (name, model) in enumerate(self.base_models.items()):
                # Create a fresh model instance for each fold
                if name == 'CatBoost':
                    fold_model = cb.CatBoostRegressor(
                        iterations=100, depth=6, learning_rate=0.1,
                        random_state=self.random_state, verbose=False
                    )
                elif name == 'LightGBM':
                    fold_model = lgb.LGBMRegressor(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=self.random_state, n_jobs=-1, verbosity=-1
                    )
                elif name == 'RandomForest':
                    fold_model = RandomForestRegressor(
                        n_estimators=100, max_depth=10,
                        random_state=self.random_state, n_jobs=-1
                    )
                elif name == 'MLP':
                    fold_model = MLPRegressor(
                        hidden_layer_sizes=(100, 50), max_iter=500,
                        random_state=self.random_state, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=10
                    )
                
                fold_model.fit(X_fold_train, y_fold_train)
                meta_features[val_idx, model_idx] = fold_model.predict(X_fold_val)
        
        # Step 2: Train meta-model on out-of-fold predictions
        print(f"\nTraining meta-model ({self.meta_model_type})...")
        self.meta_model.fit(meta_features, y_train)
        print(f"Meta-model fitted.")
        
        # Step 3: Retrain base models on full training set
        print("\nRetraining base models on full training set...")
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
            print(f"  {name} fitted.")
        
        print("\nStacking ensemble training complete!")
    
    def predict(self, X):
        """
        Make predictions using stacking ensemble.
        
        Steps:
        1. Get predictions from all base models
        2. Use base model predictions as features for meta-model
        3. Meta-model produces final prediction
        """
        X = np.array(X)
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        # Get predictions from each base model
        predictions = {}
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            pred = model.predict(X)
            predictions[name] = pred
            base_predictions[:, model_idx] = pred
        
        # Use meta-model to combine base predictions
        ensemble_pred = self.meta_model.predict(base_predictions)
        
        return ensemble_pred, predictions
    
    def update_weights(self, y_true, predictions, metric='rmse'):
        """
        Track performance and update weights using dynamic weighting.
        Concept 1: Dynamic Weighting with exponential moving average.
        """
        errors = {}
        
        for name, pred in predictions.items():
            if metric == 'rmse':
                error = np.sqrt(np.mean((y_true - pred) ** 2))
            elif metric == 'mae':
                error = np.mean(np.abs(y_true - pred))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            errors[name] = error
            if name in self.performance_history:
                self.performance_history[name].append(error)
        
        # Concept 1: Dynamic Weighting - Calculate target weights based on errors
        # Lower error = higher weight
        max_error = max(errors.values()) if errors else 1.0
        min_error = min(errors.values()) if errors else 0.0
        
        if max_error > min_error and len(errors) > 0:
            # Inverse error weighting (lower error = higher weight)
            inverse_errors = {name: max_error - error + 1e-6 
                            for name, error in errors.items()}
            total_inverse = sum(inverse_errors.values())
            
            # Update weights using exponential moving average
            # Formula: new_weight = (1 - lr) * old_weight + lr * target_weight
            for name in self.base_models.keys():
                if name in inverse_errors:
                    target_weight = inverse_errors[name] / total_inverse
                    self.weights[name] = (1 - self.learning_rate) * self.weights[name] + \
                                       self.learning_rate * target_weight
        
        return errors
    
    def calculate_confidence_score(self, name):
        """
        Concept 4: Confidence Scoring
        Calculate confidence score based on performance stability.
        Formula: confidence = 1 / (1 + CV) where CV = std(error) / mean(error)
        """
        if name not in self.performance_history or len(self.performance_history[name]) < 2:
            return 1.0
        
        errors = list(self.performance_history[name])
        
        # Lower variance = higher confidence
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        
        if error_mean > 0:
            cv = error_std / error_mean  # Coefficient of variation
            confidence = 1 / (1 + cv)  # Normalize to [0, 1]
        else:
            confidence = 1.0
        
        # Update confidence score if key exists
        if name in self.confidence_scores:
            self.confidence_scores[name] = confidence
        elif name == 'MetaModel' and 'MetaModel' not in self.confidence_scores:
            self.confidence_scores['MetaModel'] = confidence
        
        return confidence
    
    def rolling_evaluation(self, X, y, window_size=None):
        """
        Evaluate models using rolling window approach.
        Note: For stacking, this retrains the full ensemble on each window.
        """
        if window_size is None:
            window_size = self.window_size
        
        print(f"\nRolling evaluation with window size {window_size}...")
        print("Note: Stacking ensemble will be retrained on each window.")
        
        n_samples = len(X)
        results = []
        
        for i in range(window_size, n_samples):
            # Get window data
            X_window = X[i-window_size:i]
            y_window = y[i-window_size:i]
            
            # Retrain stacking ensemble on window
            # (This is computationally expensive but necessary for stacking)
            self.fit(X_window, y_window)
            
            # Predict on next sample
            if i < n_samples - 1:
                X_next = X[i:i+1]
                y_next = y[i:i+1]
                
                ensemble_pred, individual_preds = self.predict(X_next)
                
                # Track performance
                errors = self.update_weights(y_next.values, individual_preds)
                
                # Track ensemble error
                ensemble_error = np.sqrt(np.mean((y_next.values - ensemble_pred) ** 2))
                if 'MetaModel' in self.performance_history:
                    self.performance_history['MetaModel'].append(ensemble_error)
                
                # Calculate confidence scores
                for name in self.base_models.keys():
                    self.calculate_confidence_score(name)
                
                results.append({
                    'index': i,
                    'true': y_next.values[0],
                    'ensemble_pred': ensemble_pred[0],
                    'errors': errors.copy(),
                    'weights': self.weights.copy(),
                    'confidence': self.confidence_scores.copy()
                })
        
        return pd.DataFrame(results)
    
    def online_adaptation(self, X_new, y_new):
        """
        Online adaptation: track performance with new data.
        Note: Full retraining of stacking ensemble requires more data.
        """
        print("\nPerforming online adaptation...")
        
        # Get predictions
        ensemble_pred, individual_preds = self.predict(X_new)
        
        # Track performance
        errors = self.update_weights(y_new, individual_preds)
        
        # Track ensemble error
        ensemble_error = np.sqrt(np.mean((y_new - ensemble_pred) ** 2))
        if 'MetaModel' in self.performance_history:
            self.performance_history['MetaModel'].append(ensemble_error)
        
        # Update confidence scores
        for name in self.base_models.keys():
            self.calculate_confidence_score(name)
        
        # Note: Full stacking retraining would require refitting on accumulated data
        # For now, we just track performance
        
        return {
            'prediction': ensemble_pred,
            'errors': errors,
            'weights': self.weights.copy(),
            'confidence': self.confidence_scores.copy()
        }
    
    def get_ensemble_info(self):
        """Get current ensemble information."""
        info = {
            'meta_model_type': self.meta_model_type,
            'n_folds': self.n_folds,
            'weights': self.weights.copy(),  # For compatibility
            'confidence_scores': self.confidence_scores.copy(),
            'recent_performance': {}
        }
        
        for name in list(self.base_models.keys()) + ['MetaModel']:
            if name in self.performance_history and len(self.performance_history[name]) > 0:
                info['recent_performance'][name] = {
                    'mean': np.mean(self.performance_history[name]),
                    'std': np.std(self.performance_history[name]),
                    'latest': self.performance_history[name][-1]
                }
        
        return info
    
    def save_ensemble_state(self, filepath='ensemble_state.json'):
        """Save ensemble state."""
        import json
        
        state = {
            'ensemble_type': 'stacking',
            'meta_model_type': self.meta_model_type,
            'n_folds': self.n_folds,
            'weights': self.weights,  # For compatibility
            'confidence_scores': self.confidence_scores,
            'performance_history': {
                name: list(history) 
                for name, history in self.performance_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Ensemble state saved to {filepath}")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
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
    
    # Initialize stacking ensemble
    # You can change meta_model_type to 'ridge', 'xgboost', or 'neural_network'
    ensemble = AdaptiveMetaEnsemble(
        meta_model_type='ridge',  # Options: 'ridge', 'xgboost', 'neural_network'
        n_folds=5,
        window_size=50,
        learning_rate=0.1
    )
    
    # Fit stacking ensemble
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    ensemble_pred, individual_preds = ensemble.predict(X_test)
    
    # Calculate ensemble metrics
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    # Calculate individual model metrics
    individual_metrics = {}
    for name, pred in individual_preds.items():
        individual_metrics[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAE': mean_absolute_error(y_test, pred),
            'R²': r2_score(y_test, pred)
        }
    
    # Display comparison table
    print("\n" + "="*70)
    print("STACKING ENSEMBLE PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\nBase Models: CatBoost, LightGBM, RandomForest, MLP")
    print(f"Meta-Model: {ensemble.meta_model_type.upper()}")
    print(f"\n{'Model':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*70)
    
    # Display individual models
    for name, metrics in individual_metrics.items():
        print(f"{name:<15} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['R²']:<12.4f}")
    
    # Display ensemble
    print("-"*70)
    print(f"{'STACKING':<15} {ensemble_rmse:<12.4f} {ensemble_mae:<12.4f} {ensemble_r2:<12.4f}")
    print("="*70)
    
    # Calculate improvement over best individual model
    best_individual_rmse = min([m['RMSE'] for m in individual_metrics.values()])
    best_individual_mae = min([m['MAE'] for m in individual_metrics.values()])
    best_individual_r2 = max([m['R²'] for m in individual_metrics.values()])
    
    rmse_improvement = ((best_individual_rmse - ensemble_rmse) / best_individual_rmse) * 100
    mae_improvement = ((best_individual_mae - ensemble_mae) / best_individual_mae) * 100
    r2_improvement = ((ensemble_r2 - best_individual_r2) / abs(best_individual_r2)) * 100 if best_individual_r2 != 0 else 0
    
    print(f"\nStacking Improvement over Best Individual Model:")
    print(f"  RMSE: {rmse_improvement:+.2f}% (lower is better)")
    print(f"  MAE:  {mae_improvement:+.2f}% (lower is better)")
    print(f"  R²:   {r2_improvement:+.2f}% (higher is better)")
    
    # Track performance
    errors = ensemble.update_weights(y_test.values, individual_preds)
    
    # Track ensemble error
    ensemble_error = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    if 'MetaModel' in ensemble.performance_history:
        ensemble.performance_history['MetaModel'].append(ensemble_error)
    
    print(f"\nBase Model Errors (RMSE):")
    for name, error in errors.items():
        print(f"  {name}: {error:.4f}")
    print(f"  Meta-Model (Stacking): {ensemble_error:.4f}")
    
    # Calculate confidence scores
    for name in ensemble.base_models.keys():
        confidence = ensemble.calculate_confidence_score(name)
        print(f"{name} Confidence: {confidence:.4f}")
    
    # Get ensemble info
    info = ensemble.get_ensemble_info()
    print(f"\nEnsemble Info:")
    print(f"  Type: Stacking")
    print(f"  Meta-Model: {info['meta_model_type']}")
    print(f"  CV Folds: {info['n_folds']}")
    
    # Save state
    ensemble.save_ensemble_state()
    
    print("\n" + "="*50)
    print("Module 5: Stacking Ensemble Complete!")
    print("="*50)
