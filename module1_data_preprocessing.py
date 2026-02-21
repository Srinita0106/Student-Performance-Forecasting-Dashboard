"""
Module 1: Data Understanding & Preprocessing
Handles dataset loading, data dictionary creation, feature categorization,
missing value imputation, outlier detection, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, file_path):
        """Initialize the preprocessor with dataset path."""
        self.file_path = file_path
        self.df = None
        self.data_dict = {}
        self.feature_categories = {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def load_data(self):
        """Load the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def create_data_dictionary(self):
        """Create a comprehensive data dictionary."""
        print("\nCreating data dictionary...")
        self.data_dict = {
            'column_name': [],
            'data_type': [],
            'missing_count': [],
            'missing_percentage': [],
            'unique_values': [],
            'sample_values': []
        }
        
        for col in self.df.columns:
            self.data_dict['column_name'].append(col)
            self.data_dict['data_type'].append(str(self.df[col].dtype))
            missing_count = self.df[col].isnull().sum()
            self.data_dict['missing_count'].append(missing_count)
            self.data_dict['missing_percentage'].append(f"{(missing_count/len(self.df)*100):.2f}%")
            self.data_dict['unique_values'].append(self.df[col].nunique())
            sample_vals = self.df[col].dropna().head(3).tolist()
            self.data_dict['sample_values'].append(sample_vals)
        
        data_dict_df = pd.DataFrame(self.data_dict)
        data_dict_df.to_csv('data_dictionary.csv', index=False)
        print("Data dictionary saved to 'data_dictionary.csv'")
        return data_dict_df
    
    def categorize_features(self):
        """Categorize features into numerical and categorical."""
        print("\nCategorizing features...")
        
        numerical_features = []
        categorical_features = []
        
        for col in self.df.columns:
            if col == 'Exam_Score':  # Target variable
                continue
            if self.df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (low cardinality)
                if self.df[col].nunique() < 20 and self.df[col].dtype == 'int64':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        self.feature_categories = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'target': 'Exam_Score'
        }
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        return self.feature_categories
    
    def detect_outliers(self, method='iqr'):
        """Detect outliers using IQR or Z-score method."""
        print("\nDetecting outliers...")
        outlier_info = {}
        
        for col in self.feature_categories['numerical']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        outlier_df = pd.DataFrame(outlier_info).T
        outlier_df.to_csv('outlier_analysis.csv')
        print("Outlier analysis saved to 'outlier_analysis.csv'")
        return outlier_info
    
    def handle_outliers(self, method='cap'):
        """Handle outliers by capping or removing."""
        print("\nHandling outliers...")
        for col in self.feature_categories['numerical']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        print(f"Outliers handled. New shape: {self.df.shape}")
        return self.df
    
    def impute_missing_values(self, numerical_strategy='mean', categorical_strategy='most_frequent'):
        """Impute missing values."""
        print("\nImputing missing values...")
        
        # Numerical imputation
        if self.feature_categories['numerical']:
            num_imputer = SimpleImputer(strategy=numerical_strategy)
            self.df[self.feature_categories['numerical']] = num_imputer.fit_transform(
                self.df[self.feature_categories['numerical']]
            )
            self.imputers['numerical'] = num_imputer
        
        # Categorical imputation
        if self.feature_categories['categorical']:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            self.df[self.feature_categories['categorical']] = cat_imputer.fit_transform(
                self.df[self.feature_categories['categorical']]
            )
            self.imputers['categorical'] = cat_imputer
        
        print(f"Missing values imputed. Remaining missing: {self.df.isnull().sum().sum()}")
        return self.df
    
    def calculate_skewness_kurtosis(self):
        """Calculate skewness and kurtosis for numerical features."""
        print("\nCalculating skewness and kurtosis...")
        stats = {}
        
        for col in self.feature_categories['numerical']:
            stats[col] = {
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis()
            }
        
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv('skewness_kurtosis.csv')
        print("Skewness and kurtosis saved to 'skewness_kurtosis.csv'")
        return stats_df
    
    def encode_categorical(self, encoding_type='target'):
        """Encode categorical variables."""
        print(f"\nEncoding categorical variables using {encoding_type} encoding...")
        
        if encoding_type == 'target':
            # Target encoding
            target = self.df[self.feature_categories['target']]
            for col in self.feature_categories['categorical']:
                target_mean = self.df.groupby(col)[self.feature_categories['target']].mean()
                self.df[f'{col}_encoded'] = self.df[col].map(target_mean)
                self.encoders[col] = target_mean
                self.df.drop(col, axis=1, inplace=True)
        
        elif encoding_type == 'onehot':
            # One-hot encoding
            encoded_df = pd.get_dummies(self.df[self.feature_categories['categorical']], 
                                       prefix=self.feature_categories['categorical'])
            self.df = pd.concat([self.df.drop(self.feature_categories['categorical'], axis=1), 
                               encoded_df], axis=1)
        
        elif encoding_type == 'label':
            # Label encoding
            for col in self.feature_categories['categorical']:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
        
        print(f"Encoding complete. New shape: {self.df.shape}")
        return self.df
    
    def scale_features(self, scaler_type='standard'):
        """Scale numerical features."""
        print(f"\nScaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Scale numerical features
        numerical_cols = [col for col in self.feature_categories['numerical'] 
                         if col in self.df.columns]
        
        if numerical_cols:
            self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
            self.scalers['numerical'] = scaler
        
        print("Scaling complete.")
        return self.df
    
    def get_preprocessed_data(self):
        """Return the preprocessed dataset."""
        return self.df
    
    def save_preprocessed_data(self, file_path='preprocessed_data.csv'):
        """Save preprocessed data."""
        self.df.to_csv(file_path, index=False)
        print(f"\nPreprocessed data saved to '{file_path}'")


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('StudentPerformanceFactors.csv')
    
    # Load data
    df = preprocessor.load_data()
    
    # Create data dictionary
    data_dict = preprocessor.create_data_dictionary()
    
    # Categorize features
    feature_categories = preprocessor.categorize_features()
    
    # Detect outliers
    outlier_info = preprocessor.detect_outliers()
    
    # Handle outliers
    df = preprocessor.handle_outliers(method='cap')
    
    # Calculate skewness and kurtosis
    stats = preprocessor.calculate_skewness_kurtosis()
    
    # Impute missing values
    df = preprocessor.impute_missing_values()
    
    # Encode categorical variables
    df = preprocessor.encode_categorical(encoding_type='target')
    
    # Scale features
    df = preprocessor.scale_features(scaler_type='robust')
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data('preprocessed_data.csv')
    
    print("\n" + "="*50)
    print("Module 1: Data Preprocessing Complete!")
    print("="*50)
