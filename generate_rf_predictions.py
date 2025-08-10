import numpy as np
import pandas as pd
import os
import time
import sys
import traceback

# Add the current directory to the path to find the RandomForestRegressor module
sys.path.append(os.getcwd())
try:
    from random_forest_regression import RandomForestRegressor
    print("Successfully imported RandomForestRegressor")
except Exception as e:
    print(f"Error importing RandomForestRegressor: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_housing_data(file_path):
    """Load housing data from CSV file into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    print("Generating Random Forest predictions using best hyperparameters from different optimization methods...")
    
    # Load optimized transformed data
    train_file = "optimized_data/train_optimized.csv"
    test_file = "optimized_data/test_optimized.csv"
    
    train_df = load_housing_data(train_file)
    test_df = load_housing_data(test_file)
    
    if train_df is None or test_df is None:
        print("Failed to load optimized data. Trying to load original data...")
        
        # Try loading original data
        train_file = "/root/Rajnandini/test code/train_housingprices.csv"
        test_file = "/root/Rajnandini/test code/test_housingprices.csv"
        
        train_df = load_housing_data(train_file)
        test_df = load_housing_data(test_file)
        
        if train_df is None or test_df is None:
            print("Failed to load data. Please check file paths.")
            return
    
    print(f"Loaded training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    print(f"Loaded test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Check if we're using optimized data or original data
    using_optimized = 'SalePrice_log1p' in train_df.columns
    
    # Features to use for modeling
    if using_optimized:
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
        
        # Extract features and target
        X_train = train_df[model_features].fillna(0).values
        y_train = train_df['SalePrice_log1p'].values
        X_test = test_df[model_features].fillna(0).values
    else:
        # Basic preprocessing for original data
        # Remove ID column
        if 'Id' in train_df.columns:
            train_df = train_df.drop('Id', axis=1)
        if 'Id' in test_df.columns:
            test_ids = test_df['Id'].values
            test_df = test_df.drop('Id', axis=1)
        else:
            test_ids = np.arange(1, len(test_df) + 1)
            
        # Handle categorical variables
        categorical_cols = train_df.select_dtypes(include=['object']).columns
        train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
        test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
        
        # Align the training and test data to have the same columns
        common_cols = [col for col in train_encoded.columns if col in test_encoded.columns and col != 'SalePrice']
        X_train = train_encoded[common_cols].fillna(0).values
        y_train = train_encoded['SalePrice'].values
        X_test = test_encoded[common_cols].fillna(0).values
        model_features = common_cols
    
    print(f"Using {len(model_features)} features for Random Forest modeling")
    
    # Get test IDs
    if 'Id' in test_df.columns:
        test_ids = test_df['Id'].values
    else:
        test_ids = np.arange(1, len(X_test) + 1)
    
    # Best hyperparameters from each optimization method
    best_params = {
        'random_search': {
            'n_estimators': 164,
            'max_depth': 14,
            'min_samples_split': 3,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42
        },
        'grid_search': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'log2',
            'bootstrap': True,
            'random_state': 42
        },
        'bayesian': {
            'n_estimators': 152,
            'max_depth': 24,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'log2',
            'bootstrap': True,
            'random_state': 42
        }
    }
    
    # Generate predictions for each optimization method
    for method, params in best_params.items():
        print(f"\nTraining Random Forest with best parameters from {method.upper()}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Train model with best parameters
        start_time = time.time()
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Generate predictions
        start_time = time.time()
        y_pred = rf.predict(X_test)
        predict_time = time.time() - start_time
        
        # Transform predictions back to original scale if needed
        if using_optimized:
            original_preds = np.expm1(y_pred)  # inverse of log1p
        else:
            original_preds = y_pred
        
        # Save predictions to file
        output_file = f"rf_{method}_optimization_predictions.csv"
        
        with open(output_file, 'w', newline='') as file:
            import csv
            writer = csv.writer(file)
            writer.writerow(['Id', 'SalePrice'])
            
            for i, pred in enumerate(original_preds):
                writer.writerow([test_ids[i], round(pred, 4)])
        
        print(f"Saved predictions to {output_file}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Prediction time: {predict_time:.2f} seconds")
        
        # Print feature importances
        if hasattr(rf, 'feature_importances_') and rf.feature_importances_ is not None:
            print("\nTop 5 important features:")
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(5, len(model_features))):
                print(f"  {model_features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    print("\nAll Random Forest predictions generated successfully!")

if __name__ == "__main__":
    main()
