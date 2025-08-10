import numpy as np
import pandas as pd
import os
from random_forest_regression import RandomForestRegressor

def load_housing_data(file_path):
    """Load housing data from CSV file into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    print("Training Random Forest model with 200 trees and unlimited depth...")
    
    # Load optimized transformed data
    train_file = "optimized_data/train_optimized.csv"
    test_file = "optimized_data/test_optimized.csv"
    
    train_df = load_housing_data(train_file)
    test_df = load_housing_data(test_file)
    
    if train_df is None or test_df is None:
        print("Failed to load optimized data. Make sure you've run housing_optimized_transform.py first.")
        return
    
    print(f"Loaded transformed training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    print(f"Loaded transformed test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Features to use for modeling
    binary_features = [col for col in train_df.columns if col.endswith('_binary')]
    log1p_features = [col for col in train_df.columns if col.endswith('_log1p') and col != 'SalePrice_log1p']
    
    # Original moderate zero features to keep
    moderate_features = [
        'BsmtFinSF2', 'EnclosedPorch', 'HalfBath', 'MasVnrArea',
        'BsmtFullBath', '2ndFlrSF', 'WoodDeckSF', 'Fireplaces',
        'OpenPorchSF', 'BsmtFinSF1'
    ]
    
    # Other potentially useful features
    other_features = [
        'OverallQual', 'OverallCond', 'MSSubClass'
    ]
    
    # Combine all feature lists
    all_features = binary_features + log1p_features + moderate_features + other_features
    
    # Filter to only include columns that exist in both dataframes
    model_features = [f for f in all_features if f in train_df.columns and f in test_df.columns]
    
    print(f"Using {len(model_features)} features for Random Forest modeling")
    
    # Extract features and target
    X_train = train_df[model_features].fillna(0).values
    y_train = train_df['SalePrice_log1p'].values if 'SalePrice_log1p' in train_df.columns else train_df['SalePrice'].values
    X_test = test_df[model_features].fillna(0).values
    
    # Get test IDs
    test_ids = test_df['Id'].values if 'Id' in test_df.columns else np.arange(1, len(X_test) + 1)
    
    # Set parameters for the Random Forest model
    n_estimators = 200
    max_depth = None  # Unlimited depth
    
    print(f"\nTraining Random Forest model:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    
    # Create and train the model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    print("Starting model training...")
    rf_model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)
    
    # Transform predictions back to original scale
    original_preds = np.expm1(y_pred)  # inverse of log1p
    
    # Save predictions to file
    output_file = f"optimized_rf_{n_estimators}trees_depthNone_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for i, pred in enumerate(original_preds):
            writer.writerow([test_ids[i], round(pred, 4)])
    
    print(f"Saved predictions to {output_file}")
    
    # Print feature importances
    if hasattr(rf_model, 'feature_importances_'):
        print("\nTop 15 important features:")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(15, len(model_features))):
            print(f"  {model_features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    print("\nRandom Forest model with unlimited depth complete!")

if __name__ == "__main__":
    main()
