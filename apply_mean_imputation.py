import numpy as np
import pandas as pd
import os

def load_housing_data(file_path):
    """Load housing data from CSV file into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def apply_mean_imputation(train_df, test_df):
    """Apply mean imputation to numeric features"""
    print("Applying mean imputation to numeric features...")
    
    # Make copies to avoid modifying the originals
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    
    # Get numeric columns
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Calculate means from training data
    means = {}
    for col in numeric_cols:
        if col != 'Id' and col != 'SalePrice':  # Skip Id and target
            means[col] = train_df[col].mean()
    
    # Apply mean imputation to training data
    for col, mean_val in means.items():
        missing_count = train_imputed[col].isnull().sum()
        if missing_count > 0:
            print(f"  - Imputing {missing_count} missing values in {col} with mean {mean_val:.4f}")
            train_imputed[col].fillna(mean_val, inplace=True)
    
    # Apply mean imputation to test data using training means
    for col, mean_val in means.items():
        if col in test_imputed.columns:
            missing_count = test_imputed[col].isnull().sum()
            if missing_count > 0:
                print(f"  - Imputing {missing_count} missing values in test {col} with mean {mean_val:.4f}")
                test_imputed[col].fillna(mean_val, inplace=True)
    
    return train_imputed, test_imputed

def main():
    print("Applying mean imputation to original housing data...")
    
    # Load original data
    train_file_path = "train_housingprices.csv"
    test_file_path = "test_housingprices.csv"
    
    train_df = load_housing_data(train_file_path)
    test_df = load_housing_data(test_file_path)
    
    if train_df is None or test_df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    print(f"Loaded test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Apply mean imputation
    train_imputed, test_imputed = apply_mean_imputation(train_df, test_df)
    
    # Create directory for output if it doesn't exist
    if not os.path.exists('mean_imputed_data'):
        os.makedirs('mean_imputed_data')
    
    # Save imputed data
    train_imputed.to_csv('mean_imputed_data/train_mean_imputed.csv', index=False)
    test_imputed.to_csv('mean_imputed_data/test_mean_imputed.csv', index=False)
    print("\nSaved imputed data to mean_imputed_data directory")
    
    # Check if imputation was successful
    train_missing = train_imputed.isnull().sum().sum()
    test_missing = test_imputed.isnull().sum().sum()
    
    if train_missing > 0:
        print(f"WARNING: Training data still has {train_missing} missing values after imputation")
    else:
        print("Training data imputation complete - no missing values remain in numeric features")
    
    if test_missing > 0:
        print(f"WARNING: Test data still has {test_missing} missing values after imputation")
    else:
        print("Test data imputation complete - no missing values remain in numeric features")

if __name__ == "__main__":
    main()
