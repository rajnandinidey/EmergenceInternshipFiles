import numpy as np
import pandas as pd
import os
import time
import optuna
from random_forest_regression import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

# Wrapper class to make our custom RandomForestRegressor compatible with scikit-learn's cross_val_score
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

def objective(trial, X_train, y_train, param_grid):
    """Define the objective function for Optuna to minimize using grid search approach"""
    
    # Get hyperparameters from grid
    n_estimators = trial.suggest_categorical('n_estimators', param_grid['n_estimators'])
    max_depth = trial.suggest_categorical('max_depth', param_grid['max_depth'])
    min_samples_split = trial.suggest_categorical('min_samples_split', param_grid['min_samples_split'])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', param_grid['min_samples_leaf'])
    max_features = trial.suggest_categorical('max_features', param_grid['max_features'])
    
    # Create model with suggested hyperparameters
    rf_wrapper = RandomForestRegressorWrapper(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=42
    )
    
    # Create KFold cross-validator
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Calculate cross-validation score (negative MSE)
    cv_scores = cross_val_score(
        rf_wrapper, 
        X_train, 
        y_train, 
        scoring='neg_mean_squared_error',
        cv=kfold,
        n_jobs=2  # Limit to 2 cores to avoid resource issues
    )
    
    # Return the mean negative MSE (we want to maximize this, which is equivalent to minimizing MSE)
    return np.mean(cv_scores)

def main():
    print("Performing Optuna Grid Search for Random Forest hyperparameter tuning...")
    
    # Record start time for the entire process
    total_start_time = time.time()
    
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
    
    # Define parameter grid for grid search (reduced size)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    print("Using a reduced parameter grid for faster execution:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Calculate total number of combinations
    total_combinations = (
        len(param_grid['n_estimators']) * 
        len(param_grid['max_depth']) * 
        len(param_grid['min_samples_split']) * 
        len(param_grid['min_samples_leaf']) * 
        len(param_grid['max_features'])
    )
    
    print(f"\nGrid Search will evaluate {total_combinations} combinations of hyperparameters")
    
    # Create Optuna study for grid search
    print("\nStarting Optuna Grid Search for hyperparameter tuning...")
    
    # Record start time for optimization
    optim_start_time = time.time()
    
    # Create study with GridSampler
    search_space = {
        'n_estimators': param_grid['n_estimators'],
        'max_depth': param_grid['max_depth'],
        'min_samples_split': param_grid['min_samples_split'],
        'min_samples_leaf': param_grid['min_samples_leaf'],
        'max_features': param_grid['max_features']
    }
    
    study = optuna.create_study(
        direction='maximize',  # We want to maximize the negative MSE
        sampler=optuna.samplers.GridSampler(search_space)
    )
    
    # Run optimization
    print("\nStarting optimization...")
    try:
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, param_grid),
            n_jobs=1  # Run sequentially to avoid resource issues
        )
        print(f"Optimization completed successfully with {len(study.trials)} trials")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    # Calculate optimization time
    optim_time = time.time() - optim_start_time
    
    # Print best parameters and score
    print("\nOptuna Grid Search Results:")
    print(f"Best parameters: {study.best_params}")
    print(f"Best CV score (neg MSE): {study.best_value:.4f}")
    print(f"Best RMSE: {np.sqrt(-study.best_value):.4f}")
    print(f"Optimization completed in {optim_time:.2f} seconds")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_rf = RandomForestRegressorWrapper(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        bootstrap=True,
        random_state=42
    )
    
    # Fit the model
    best_rf.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = best_rf.predict(X_test)
    
    # Transform predictions back to original scale if needed
    if using_optimized:
        original_preds = np.expm1(y_pred)  # inverse of log1p
    else:
        original_preds = y_pred
    
    # Save predictions to file
    best_params_str = "_".join([f"{k}_{v}" for k, v in study.best_params.items() if v is not None])
    output_file = f"rf_optuna_gridsearch_{best_params_str}_predictions.csv"
    
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
        
        # Create a DataFrame for better visualization and saving
        feature_importance_df = pd.DataFrame({
            'Feature': [model_features[i] for i in indices[:20]],
            'Importance': importances[indices[:20]]
        })
        
        # Save feature importances to CSV
        feature_importance_df.to_csv('rf_gridsearch_feature_importances.csv', index=False)
        print(f"Feature importances saved to rf_gridsearch_feature_importances.csv")
        
        # Print top 10
        for i in range(min(10, len(model_features))):
            print(f"  {model_features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save all trial results to CSV
    trials_df = pd.DataFrame([
        {
            **trial.params,
            'value': trial.value,
            'rmse': np.sqrt(-trial.value) if trial.value is not None else None
        }
        for trial in study.trials
    ])
    
    # Sort by performance (best first)
    trials_df = trials_df.sort_values('value', ascending=False)
    
    # Save to CSV
    trials_df.to_csv('rf_gridsearch_trials.csv', index=False)
    print(f"\nSaved all trial results to rf_gridsearch_trials.csv")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    
    # Extract values and iteration numbers
    values = [t.value for t in study.trials if t.value is not None]
    iterations = range(1, len(values) + 1)
    
    # Plot the optimization history
    plt.plot(iterations, [-v for v in values], marker='o', linestyle='-', markersize=4)
    plt.axhline(y=-study.best_value, color='r', linestyle='--', label=f'Best RMSE: {np.sqrt(-study.best_value):.4f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Optuna Grid Search Optimization History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('rf_gridsearch_optimization_history.png')
    print("Saved optimization history plot to rf_gridsearch_optimization_history.png")
    
    # Calculate total execution time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Save execution time to a file
    with open('rf_gridsearch_execution_time.txt', 'w') as f:
        f.write(f"Optuna Grid Search Optimization Execution Time\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Number of combinations: {total_combinations}\n")
        f.write(f"Optimization time: {optim_time:.2f} seconds\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n")
        f.write(f"Average time per combination: {optim_time/total_combinations:.2f} seconds\n")
    
    print(f"Execution time details saved to rf_gridsearch_execution_time.txt")
    
    print("\nRandom Forest Optuna Grid Search hyperparameter tuning complete!")

if __name__ == "__main__":
    main()
