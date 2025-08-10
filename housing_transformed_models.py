import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from lasso_regression import LassoRegression
from ridge_regression import RidgeRegression
from decision_tree_regression import DecisionTreeRegressor

def load_data(file_path):
    """Load data from CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def prepare_features(df, target_col=None):
    """Prepare features for modeling, handling missing values"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Get numeric columns only
    numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Remove target column from features if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Fill missing values with column means
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    # Extract features and convert to numpy arrays
    X = df_copy[numeric_cols].values
    
    # Extract target if specified
    y = None
    if target_col and target_col in df_copy.columns:
        y = df_copy[target_col].values
    
    return X, y, numeric_cols

def train_lasso_models(X_train, y_train, alphas):
    """Train Lasso regression models with different alphas"""
    models = {}
    for alpha in alphas:
        print(f"Training Lasso model with alpha={alpha}...")
        model = LassoRegression(alpha=alpha, max_iter=10000, tol=1e-6)
        model.fit(X_train, y_train)
        models[alpha] = model
    return models

def train_ridge_models(X_train, y_train, alphas):
    """Train Ridge regression models with different alphas"""
    models = {}
    for alpha in alphas:
        print(f"Training Ridge model with alpha={alpha}...")
        model = RidgeRegression(alpha=alpha, solver='closed_form')
        model.fit(X_train, y_train)
        models[alpha] = model
    return models

def train_polynomial_models(X_train, y_train, degrees, alpha=0.1):
    """Train polynomial regression models with different degrees"""
    models = {}
    for degree in degrees:
        print(f"Training Polynomial model with degree={degree}...")
        # Transform features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_train)
        
        # Use Ridge regression for polynomial features to prevent overfitting
        model = RidgeRegression(alpha=alpha, solver='closed_form')
        model.fit(X_poly, y_train)
        
        models[degree] = {
            'model': model,
            'poly': poly
        }
    return models

def train_decision_tree_models(X_train, y_train, max_depths):
    """Train decision tree models with different max depths"""
    models = {}
    for max_depth in max_depths:
        print(f"Training Decision Tree model with max_depth={max_depth}...")
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        models[max_depth] = model
    return models

def evaluate_models(models, X_test, y_test, model_type, transform_func=None):
    """Evaluate models and return performance metrics"""
    results = {}
    
    for param, model in models.items():
        # Make predictions
        if model_type == 'polynomial':
            poly = model['poly']
            X_test_poly = poly.transform(X_test)
            y_pred = model['model'].predict(X_test_poly)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[param] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"{model_type.capitalize()} model with param={param}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    return results

def save_predictions(predictions, ids, model_type, param, transform_func=None):
    """Save predictions to CSV file"""
    output_file = f"{model_type}_predictions_{param}.csv"
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for i, pred in enumerate(predictions):
            # Transform prediction back to original scale if needed
            if transform_func:
                pred = transform_func(pred)
            writer.writerow([ids[i], round(pred, 4)])
    
    print(f"Predictions saved to {output_file}")
    return output_file

def create_stacked_predictions(prediction_files, ids, transform_func=None):
    """Create stacked predictions from multiple models"""
    all_predictions = {}
    
    # Load predictions from each file
    for file_path in prediction_files:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if len(row) >= 2:
                    id_val = int(row[0])
                    pred = float(row[1])
                    
                    if id_val not in all_predictions:
                        all_predictions[id_val] = []
                    all_predictions[id_val].append(pred)
    
    # Calculate average predictions
    final_predictions = {}
    for id_val, preds in all_predictions.items():
        if preds:
            final_predictions[id_val] = sum(preds) / len(preds)
    
    # Save stacked predictions
    output_file = "transformed_stacked_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for id_val in sorted(final_predictions.keys()):
            pred = final_predictions[id_val]
            # Transform prediction back to original scale if needed
            if transform_func:
                pred = transform_func(pred)
            writer.writerow([id_val, round(pred, 4)])
    
    print(f"Stacked predictions saved to {output_file}")
    return output_file

def main():
    print("Training models on log-transformed housing data...")
    
    # Load transformed data
    train_file = "transformed_data/train_housingprices_log_transformed.csv"
    test_file = "transformed_data/test_housingprices_log_transformed.csv"
    
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    if train_df is None:
        print("Failed to load training data. Exiting.")
        return
    
    if test_df is None:
        print("Failed to load test data. Exiting.")
        return
    
    print(f"Loaded training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    print(f"Loaded test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Prepare features and target
    X_train, y_train, feature_cols = prepare_features(train_df, target_col='SalePrice')
    X_test, _, _ = prepare_features(test_df)
    
    # Get test IDs
    test_ids = test_df['Id'].values
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define model parameters
    lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    ridge_alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    poly_degrees = [1, 2, 3]
    tree_depths = [5, 10, 15, 20]
    
    # Function to transform predictions back to original scale
    inverse_transform = lambda x: np.expm1(x)  # exp(x) - 1
    
    # Train models
    print("\n--- Training Lasso Models ---")
    lasso_models = train_lasso_models(X_train, y_train, lasso_alphas)
    
    print("\n--- Training Ridge Models ---")
    ridge_models = train_ridge_models(X_train, y_train, ridge_alphas)
    
    print("\n--- Training Polynomial Models ---")
    poly_models = train_polynomial_models(X_train, y_train, poly_degrees)
    
    print("\n--- Training Decision Tree Models ---")
    tree_models = train_decision_tree_models(X_train, y_train, tree_depths)
    
    # Generate predictions
    prediction_files = []
    
    print("\n--- Generating Predictions ---")
    
    # Lasso predictions
    for alpha, model in lasso_models.items():
        y_pred = model.predict(X_test)
        file_path = save_predictions(y_pred, test_ids, "lasso_transformed", f"alpha_{alpha}", inverse_transform)
        prediction_files.append(file_path)
    
    # Ridge predictions
    for alpha, model in ridge_models.items():
        y_pred = model.predict(X_test)
        file_path = save_predictions(y_pred, test_ids, "ridge_transformed", f"alpha_{alpha}", inverse_transform)
        prediction_files.append(file_path)
    
    # Polynomial predictions
    for degree, model_data in poly_models.items():
        model = model_data['model']
        poly = model_data['poly']
        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)
        file_path = save_predictions(y_pred, test_ids, "poly_transformed", f"degree_{degree}", inverse_transform)
        prediction_files.append(file_path)
    
    # Decision Tree predictions
    for depth, model in tree_models.items():
        y_pred = model.predict(X_test)
        file_path = save_predictions(y_pred, test_ids, "tree_transformed", f"depth_{depth}", inverse_transform)
        prediction_files.append(file_path)
    
    # Create stacked predictions
    print("\n--- Creating Stacked Predictions ---")
    stacked_file = create_stacked_predictions(prediction_files, test_ids, inverse_transform)
    
    print("\nAll models trained and predictions generated successfully!")

if __name__ == "__main__":
    main()
