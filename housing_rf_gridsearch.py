import numpy as np
import pandas as pd
import os
from random_forest_regression import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator

# Wrapper class to make our custom RandomForestRegressor compatible with scikit-learn's GridSearchCV
class RandomForestRegressorWrapper(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model = None
        
    def fit(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None

def load_housing_data(file_path):
    """Load housing data from CSV file into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    print("Performing GridSearchCV for Random Forest hyperparameter tuning...")
    
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
    
    # Define a balanced parameter grid - more comprehensive but still efficient
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create the wrapper model
    rf_wrapper = RandomForestRegressorWrapper(random_state=42)
    
    # Create KFold cross-validator
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Starting GridSearchCV for Random Forest hyperparameter tuning...")
    print(f"Parameter grid: {param_grid}")
    print(f"Cross-validation: {kfold.n_splits}-fold CV")
    
    # Create GridSearchCV object with more efficient search strategy
    grid_search = GridSearchCV(
        estimator=rf_wrapper,
        param_grid=param_grid,
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all available cores
        verbose=2,
        error_score='raise'
    )
    
    # Fit GridSearchCV
    print("\nFitting GridSearchCV. This may take some time...")
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    
    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)
    
    # Transform predictions back to original scale
    original_preds = np.expm1(y_pred)  # inverse of log1p
    
    # Save predictions to file
    best_params_str = "_".join([f"{k}_{v}" for k, v in grid_search.best_params_.items() if v is not None])
    output_file = f"optimized_rf_gridsearch_{best_params_str}_predictions.csv"
    
    with open(output_file, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for i, pred in enumerate(original_preds):
            writer.writerow([test_ids[i], round(pred, 4)])
    
    print(f"\nSaved predictions from best model to {output_file}")
    
    # Print feature importances
    if hasattr(best_rf, 'feature_importances_') and best_rf.feature_importances_ is not None:
        print("\nTop 10 important features from best model:")
        importances = best_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(model_features))):
            print(f"  {model_features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save all CV results to CSV for further analysis
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Sort results by mean test score (descending)
    cv_results_df = cv_results_df.sort_values('mean_test_score', ascending=False)
    
    # Save to CSV
    cv_results_file = 'rf_gridsearch_cv_results_expanded.csv'
    cv_results_df.to_csv(cv_results_file, index=False)
    print(f"\nSaved all CV results to {cv_results_file}")
    
    # Print top 5 parameter combinations
    print("\nTop 5 parameter combinations:")
    top_params = cv_results_df[['params', 'mean_test_score', 'std_test_score']].head(5)
    for i, row in top_params.iterrows():
        params = row['params']
        score = row['mean_test_score']
        std = row['std_test_score']
        print(f"  {i+1}. {params} - RMSE: {np.sqrt(-score):.4f} (±{np.sqrt(std):.4f})")
    
    print("\nRandom Forest GridSearch hyperparameter tuning complete!")

if __name__ == "__main__":
    main()
