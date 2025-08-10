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
        if col != 'Id' and col != 'SalePrice' and col != 'SalePrice_log1p':  # Skip Id and target
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
    print("Generating Random Forest predictions using GridSearch optimized parameters with mean imputation...")
    
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
    
    # Apply mean imputation instead of filling with zeros
    train_imputed, test_imputed = apply_mean_imputation(train_df, test_df)
    
    # Features to use for modeling
    binary_features = [col for col in train_imputed.columns if col.endswith('_binary')]
    log1p_features = [col for col in train_imputed.columns if col.endswith('_log1p') and col != 'SalePrice_log1p']
    
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
    model_features = [f for f in all_features if f in train_imputed.columns and f in test_imputed.columns]
    
    print(f"Using {len(model_features)} features for Random Forest modeling")
    
    # Extract features and target
    X_train = train_imputed[model_features].values
    y_train = train_imputed['SalePrice_log1p'].values if 'SalePrice_log1p' in train_imputed.columns else train_imputed['SalePrice'].values
    X_test = test_imputed[model_features].values
    
    # Get test IDs
    test_ids = test_imputed['Id'].values if 'Id' in test_imputed.columns else np.arange(1, len(X_test) + 1)
    
    # Use the best parameters found by GridSearchCV
    print("\nTraining Random Forest model with GridSearch optimized parameters:")
    print("  n_estimators: 100")
    print("  max_depth: 15")
    print("  min_samples_split: 2")
    print("  min_samples_leaf: 1")
    print("  max_features: sqrt")
    print("  bootstrap: True")
    
    # Create and train the model with GridSearch optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = rf_model.predict(X_test)
    
    # Transform predictions back to original scale
    original_preds = np.expm1(y_pred)  # inverse of log1p
    
    # Save predictions to file
    output_file = "optimized_rf_gridsearch_mean_imputed_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for i, pred in enumerate(original_preds):
            writer.writerow([test_ids[i], round(pred, 4)])
    
    print(f"Saved predictions to {output_file}")
    
    # Print feature importances
    if hasattr(rf_model, 'feature_importances_'):
        print("\nTop 10 important features:")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(model_features))):
            print(f"  {model_features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Create stacked predictions including this Random Forest model
    print("\nUpdating stacked predictions to include mean-imputed Random Forest model...")
    
    # Load existing stacked predictions file
    existing_stacked_file = "optimized_stacked_with_rf_predictions.csv"
    if not os.path.exists(existing_stacked_file):
        existing_stacked_file = "optimized_stacked_predictions.csv"
    
    existing_predictions = {}
    
    if os.path.exists(existing_stacked_file):
        with open(existing_stacked_file, 'r') as file:
            import csv
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if len(row) >= 2:
                    id_val = int(row[0])
                    pred = float(row[1])
                    existing_predictions[id_val] = [pred]
    
    # Add mean-imputed Random Forest predictions
    with open(output_file, 'r') as file:
        import csv
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        
        for row in csv_reader:
            if len(row) >= 2:
                id_val = int(row[0])
                pred = float(row[1])
                
                if id_val not in existing_predictions:
                    existing_predictions[id_val] = []
                existing_predictions[id_val].append(pred)
    
    # Calculate average predictions
    final_predictions = {}
    for id_val, preds in existing_predictions.items():
        if preds:
            final_predictions[id_val] = sum(preds) / len(preds)
    
    # Save updated stacked predictions
    output_file = "optimized_stacked_with_rf_mean_imputed_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for id_val in sorted(final_predictions.keys()):
            pred = final_predictions[id_val]
            writer.writerow([id_val, round(pred, 4)])
    
    print(f"Updated stacked predictions saved to {output_file}")
    print("\nMean-imputed Random Forest prediction process complete!")

if __name__ == "__main__":
    main()
