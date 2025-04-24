"""
Data Preprocessing Module

This module handles the preprocessing of credit and census datasets for machine learning tasks.
It includes data cleaning, feature engineering, and standardization procedures.

Key Data Transformation Steps:
1. Data Loading & Cleaning
   - Handle missing values
   - Correct invalid data (e.g., negative ages)
   - Remove outliers if necessary

2. Feature Engineering
   - Convert categorical variables to numerical format using:
     a) Label Encoding: Converts categories to numbers (0,1,2...)
     b) One-Hot Encoding: Creates binary columns for each category
   - Scale numerical features using StandardScaler

3. Data Splitting
   - Split data into training and testing sets
   - Save processed data for model training

Author: Furukawa
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
import os

class CreditDataPreprocessor:
    """Handles preprocessing of credit data.
    
    The credit dataset contains both numerical and categorical features.
    Main preprocessing steps:
    1. Handle negative ages (data error)
    2. Fill missing values with mean
    3. Scale numerical features for better model performance
    """
    
    def __init__(self, file_path='../../data/raw/credit_data.csv'):
        """Initialize with data file path."""
        self.data = pd.read_csv(file_path)
        self.X = None
        self.y = None
        self._validate_input_data()
    
    def _validate_input_data(self):
        """Validate input data structure and content.
        
        Raises:
            ValueError: If required columns are missing or data types are incorrect
        """
        required_columns = ['age', 'income', 'loan', 'default']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Validate data types
        if not pd.api.types.is_numeric_dtype(self.data['age']):
            raise ValueError("Age column must be numeric")
        if not pd.api.types.is_numeric_dtype(self.data['income']):
            raise ValueError("Income column must be numeric")
        if not pd.api.types.is_numeric_dtype(self.data['loan']):
            raise ValueError("Loan column must be numeric")
            
        # Validate value ranges
        if self.data['income'].min() < 0:
            raise ValueError("Income values cannot be negative")
        if self.data['loan'].min() < 0:
            raise ValueError("Loan values cannot be negative")
            
        # Validate target variable
        if not set(self.data['default'].unique()).issubset({0, 1}):
            raise ValueError("Default column must contain only binary values (0 or 1)")
    
    def handle_missing_values(self):
        """Fill missing values in the dataset.
        
        For numerical features like age, we use mean imputation.
        This helps maintain the data distribution while filling gaps.
        """
        self.data['age'] = self.data['age'].fillna(self.data['age'].mean())
    
    def correct_negative_ages(self):
        """Replace negative age values with mean age.
        
        Negative ages are clearly data errors and need to be corrected.
        We replace them with the mean age of valid entries to maintain
        data distribution.
        """
        mean_age = self.data['age'][self.data['age'] > 0].mean()
        self.data.loc[self.data['age'] < 0, 'age'] = mean_age
    
    def show_data_info(self):
        """Display basic information about the dataset."""
        print("\nCredit Dataset Information:")
        print("-" * 50)
        print(self.data.info())
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Show distribution of default values
        default_counts = np.unique(self.data['default'], return_counts=True)
        print("\nDefault Distribution:")
        print(f"Non-default: {default_counts[1][0]}")
        print(f"Default: {default_counts[1][1]}")
    
    def visualize_data(self):
        """Create various visualizations for the dataset."""
        # Age distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['age'], bins=30)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()
        
        # Default distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.data['default'])
        plt.title('Default Distribution')
        plt.show()
        
        # Scatter matrix
        scatter_matrix = px.scatter_matrix(
            self.data, 
            dimensions=['income', 'age', 'loan'], 
            color='default',
            title='Feature Relationships'
        )
        return scatter_matrix
    
    def prepare_features(self):
        """Prepare features for modeling.
        
        Steps:
        1. Extract relevant columns (excluding client ID)
        2. Scale numerical features using StandardScaler
           - Standardization: (x - mean) / std_dev
           - This ensures all features are on same scale
           - Helps prevent larger-scale features from dominating
        """
        # Extract features and target
        self.X = self.data.iloc[:, 1:4].values  # income, age, loan
        self.y = self.data.iloc[:, 4].values    # default status
        
        # Standardize features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        return self.X, self.y

class CensusDataPreprocessor:
    """Handles preprocessing of census data.
    
    The census dataset contains many categorical variables that need
    special handling before they can be used in ML models.
    
    Key preprocessing steps:
    1. Convert categorical variables to numbers using Label Encoding
    2. Convert Label Encoded variables to One-Hot Encoding
       - This prevents ordinal relationships in categorical data
    3. Scale the final features using StandardScaler
    """
    
    def __init__(self, file_path='../../data/raw/census_data.csv'):
        """Initialize with data file path."""
        self.data = pd.read_csv(file_path)
        print("\nAvailable columns in census data:")
        print(self.data.columns.tolist())
        self.X = None
        self.y = None
        
        # Initialize label encoders for each categorical column
        self.label_encoders = {
            'workclass': LabelEncoder(),
            'education': LabelEncoder(),
            'marital-status': LabelEncoder(),
            'occupation': LabelEncoder(),
            'relationship': LabelEncoder(),
            'race': LabelEncoder(),
            'sex': LabelEncoder(),
            'native-country': LabelEncoder()
        }
        self._validate_input_data()
    
    def _validate_input_data(self):
        """Validate census data structure and content.
        
        Raises:
            ValueError: If required columns are missing or data format is incorrect
        """
        # Check for required columns
        required_columns = list(self.label_encoders.keys()) + ['age', 'income']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate age column
        if not pd.api.types.is_numeric_dtype(self.data['age']):
            raise ValueError("Age column must be numeric")
        if (self.data['age'] < 0).any():
            raise ValueError("Age values cannot be negative")
        if (self.data['age'] > 120).any():
            raise ValueError("Age values seem unrealistic (>120)")
            
        # Validate categorical columns have no missing values
        for col in self.label_encoders.keys():
            if self.data[col].isnull().any():
                raise ValueError(f"Column {col} contains missing values")
                
        # Validate income column format (should be binary <=50K, >50K)
        valid_income = {' <=50K', ' >50K'}
        if not set(self.data['income'].unique()).issubset(valid_income):
            raise ValueError("Income column must contain only '<=50K' or '>50K' values")
    
    def show_data_info(self):
        """Display basic information about the dataset."""
        print("\nCensus Dataset Information:")
        print("-" * 50)
        print(self.data.info())
        print("\nBasic Statistics:")
        print(self.data.describe())
    
    def visualize_data(self):
        """Create visualizations for census data."""
        # Income distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.data['income'])
        plt.title('Income Distribution')
        plt.show()
        
        # Age distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['age'], bins=30)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()
        
        # Category relationships
        category_plot = px.parallel_categories(
            self.data, 
            dimensions=['workclass', 'occupation', 'income'],
            title='Category Relationships'
        )
        return category_plot
    
    def encode_categorical_features(self):
        """Encode categorical variables using Label Encoding and One-Hot Encoding.
        
        Two-step process:
        1. Label Encoding: Convert categories to numbers
           - Each unique category gets a number
           - Problem: Creates false ordinal relationships
           
        2. One-Hot Encoding: Convert numbers to binary columns
           - Creates a new column for each category
           - Each row has 1 in its category column, 0 in others
           - Solves the ordinal relationship problem
           - Example: 
             'Color' with values 'Red'(0), 'Blue'(1), 'Green'(2) becomes:
             'Color_Red'  'Color_Blue'  'Color_Green'
                  1           0             0          # Red
                  0           1             0          # Blue
                  0           0             1          # Green
        """
        self.X = self.data.iloc[:, 0:14].values
        self.y = self.data.iloc[:, 14].values
        
        # Step 1: Apply label encoding
        categorical_columns = {
            1: 'workclass',
            3: 'education',
            5: 'marital-status',
            6: 'occupation',
            7: 'relationship',
            8: 'race',
            9: 'sex',
            13: 'native-country'
        }
        
        for col, name in categorical_columns.items():
            self.X[:, col] = self.label_encoders[name].fit_transform(self.X[:, col])
        
        # Step 2: Apply one-hot encoding
        # This creates binary columns for each category
        onehotencoder = ColumnTransformer(
            transformers=[
                ("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])
            ],
            remainder='passthrough'  # Keep other columns as is
        )
        self.X = onehotencoder.fit_transform(self.X).toarray()
        
        # Step 3: Scale all features
        # After one-hot encoding, we scale everything to ensure
        # all features contribute equally to the model
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        return self.X, self.y

class CreditRiskPreprocessor:
    """Handles preprocessing of credit risk data.
    
    This dataset contains categorical risk factors that need
    to be encoded before modeling.
    """
    
    def __init__(self, file_path='../../data/raw/credit_risk.csv'):
        """Initialize with data file path."""
        self.data = pd.read_csv(file_path)
        self.X = None
        self.y = None
        
        # Initialize label encoders for each categorical feature
        self.label_encoders = {
            'historia': LabelEncoder(),
            'divida': LabelEncoder(),
            'garantia': LabelEncoder(),
            'renda': LabelEncoder()
        }
    
    def prepare_features(self):
        """Prepare features for modeling.
        
        For this dataset, we only use Label Encoding since:
        1. The categories might have natural ordinal relationships
        2. The dataset is smaller, so One-Hot Encoding might create
           too many sparse features
        """
        self.X = self.data.iloc[:, 0:4].values
        self.y = self.data.iloc[:, 4].values
        
        # Encode categorical variables
        for idx, (name, encoder) in enumerate(self.label_encoders.items()):
            self.X[:, idx] = encoder.fit_transform(self.X[:, idx])
        
        return self.X, self.y

def save_preprocessed_data(X_train, y_train, X_test, y_test, filename):
    """Save preprocessed data to pickle file in the data/processed directory."""
    # Create the data/processed directory if it doesn't exist
    os.makedirs('../../data/processed', exist_ok=True)
    
    # Construct the full path for saving the file
    save_path = os.path.join('../../data/processed', filename)
    
    with open(save_path, 'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test], f)
    print(f"Saved preprocessed data to {save_path}")

def main():
    """Main execution function."""
    # Process Credit Data
    print("Processing Credit Data...")
    credit_processor = CreditDataPreprocessor()
    credit_processor.correct_negative_ages()
    credit_processor.handle_missing_values()
    X_credit, y_credit = credit_processor.prepare_features()
    
    # Process Census Data
    print("\nProcessing Census Data...")
    census_processor = CensusDataPreprocessor()
    X_census, y_census = census_processor.encode_categorical_features()
    
    # Process Credit Risk Data
    print("\nProcessing Credit Risk Data...")
    risk_processor = CreditRiskPreprocessor()
    X_risk, y_risk = risk_processor.prepare_features()
    
    # Split datasets
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
        X_credit, y_credit, test_size=0.25, random_state=0
    )
    
    X_census_train, X_census_test, y_census_train, y_census_test = train_test_split(
        X_census, y_census, test_size=0.15, random_state=0
    )
    
    X_risk_train, X_risk_test, y_risk_train, y_risk_test = train_test_split(
        X_risk, y_risk, test_size=0.25, random_state=0
    )

    # Save preprocessed data
    print("\nSaving preprocessed data...")
    save_preprocessed_data(
        X_credit_train, y_credit_train, X_credit_test, y_credit_test,
        'credit.pkl'
    )
    save_preprocessed_data(
        X_census_train, y_census_train, X_census_test, y_census_test,
        'census.pkl'
    )
    save_preprocessed_data(
        X_risk_train, y_risk_train, X_risk_test, y_risk_test,
        'risk.pkl'
    )
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()

