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
    print("Training Random Forest model on mean-imputed data...")
    
    # Load mean-imputed data
    train_file_path = "mean_imputed_data/train_mean_imputed.csv"
    test_file_path = "mean_imputed_data/test_mean_imputed.csv"
    
    train_df = load_housing_data(train_file_path)
    test_df = load_housing_data(test_file_path)
    
    if train_df is None or test_df is None:
        print("Failed to load imputed data. Make sure you've run apply_mean_imputation.py first.")
        return
    
    print(f"Loaded imputed training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    print(f"Loaded imputed test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Get numeric features for modeling (exclude Id and SalePrice)
    numeric_features = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Id' in numeric_features:
        numeric_features.remove('Id')
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')
    
    print(f"\nUsing {len(numeric_features)} numeric features for Random Forest modeling:")
    for i, feature in enumerate(numeric_features):
        print(f"  {i+1}. {feature}")
    
    # Extract features and target
    X_train = train_df[numeric_features].values
    y_train = train_df['SalePrice'].values
    X_test = test_df[numeric_features].values
    
    # Get test IDs
    test_ids = test_df['Id'].values
    
    # Train Random Forest models with different parameters
    rf_params = [
        {"n_estimators": 100, "max_depth": 15, "name": "rf_100trees_depth15"},
        {"n_estimators": 200, "max_depth": None, "name": "rf_200trees_depthNone"}
    ]
    
    for params in rf_params:
        print(f"\nTraining Random Forest model: {params['name']}")
        print(f"  n_estimators: {params['n_estimators']}")
        print(f"  max_depth: {params['max_depth']}")
        
        # Create and train the model
        rf_model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
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
        
        # Save predictions to file
        output_file = f"mean_imputed_{params['name']}_predictions.csv"
        with open(output_file, 'w', newline='') as file:
            import csv
            writer = csv.writer(file)
            writer.writerow(['Id', 'SalePrice'])
            
            for i, pred in enumerate(y_pred):
                writer.writerow([test_ids[i], round(pred, 4)])
        
        print(f"Saved predictions to {output_file}")
        
        # Print feature importances
        if hasattr(rf_model, 'feature_importances_'):
            print("\nTop 15 important features:")
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in range(min(15, len(numeric_features))):
                feature_idx = indices[i]
                print(f"  {numeric_features[feature_idx]}: {importances[feature_idx]:.4f}")

    print("\nRandom Forest models on mean-imputed data complete!")

if __name__ == "__main__":
    main()
