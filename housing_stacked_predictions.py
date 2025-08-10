import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from polynomial_regression import PolynomialRegression
from ridge_regression import RidgeRegression

def load_housing_data(file_path, feature_cols=['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual'], target_col='SalePrice'):
    
    #Load housing data from CSV file
    
    #Parameters:
    #file_path : path to the CSV file
    #feature_cols : list of feature column names
    #target_col : name of the target column (None for test data without target)
    
    #Returns:
    #data : pandas DataFrame containing the data
    #feature_matrix : numpy array of features
    #target_values : numpy array of target values (None for test data without target)
    #ids : list of ID values
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Extract IDs
    ids = data['Id'].values
    
    # Check if all feature columns exist
    for col in feature_cols:
        if col not in data.columns:
            raise ValueError(f"Feature column '{col}' not found in the data")
    
    # Extract features
    feature_matrix = data[feature_cols].values
    
    # Extract target if available
    target_values = None
    if target_col in data.columns:
        target_values = data[target_col].values
    
    return data, feature_matrix, target_values, ids

def normalize_data(X, X_mean=None, X_std=None):
    
    #Normalize features to prevent numerical issues
    
    #Parameters:
    #X : numpy array of feature values
    #X_mean : means for normalization (calculated if None)
    #X_std : standard deviations for normalization (calculated if None)
    
    #Returns:
    #X_norm : normalized features
    #X_mean : means used for normalization
    #X_std : standard deviations used for normalization
    
    if X_mean is None or X_std is None:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.maximum(X_std, 1e-8)  # Prevent division by zero
    
    X_norm = (X - X_mean) / X_std
    
    return X_norm, X_mean, X_std

def denormalize_predictions(y_pred_norm, y_mean, y_std):
    
    #Denormalize predictions back to original scale
    
    #Parameters:
    #y_pred_norm : normalized predictions
    #y_mean : mean of target values
    #y_std : standard deviation of target values
    
    #Returns:
    #y_pred : predictions in original scale
    
    return y_pred_norm * y_std + y_mean

def load_polynomial_model_params(file_path):
    
    #Load polynomial model parameters from CSV file
    
    #Parameters:
    #file_path : path to the CSV file
    
    #Returns:
    #params : dictionary of model parameters
    
    params = {}
    feature_names = []
    X_means = []
    X_stds = []
    coefficients = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            param_name, param_value = row
            
            if param_name == 'degree':
                params['degree'] = int(param_value)
            elif param_name == 'y_mean':
                params['y_mean'] = float(param_value)
            elif param_name == 'y_std':
                params['y_std'] = float(param_value)
            elif param_name == 'target_col':
                params['target_col'] = param_value
            elif param_name.startswith('feature_name_'):
                feature_names.append(param_value)
            elif param_name.startswith('X_mean_'):
                X_means.append(float(param_value))
            elif param_name.startswith('X_std_'):
                X_stds.append(float(param_value))
            elif param_name.startswith('coefficient_'):
                coefficients.append(float(param_value))
    
    params['feature_names'] = feature_names
    params['X_means'] = np.array(X_means)
    params['X_stds'] = np.array(X_stds)
    params['coefficients'] = np.array(coefficients)
    
    return params

def generate_polynomial_features(X, degree):
    
    #Generate polynomial features up to the specified degree
    
    #Parameters:
    #X : numpy array of shape (n_samples, n_features)
    #degree : degree of polynomial features
    
    #Returns:
    #X_poly : numpy array with polynomial features
    
    n_samples, n_features = X.shape
    n_poly_features = 1  # Intercept
    
    for d in range(1, degree + 1):
        n_poly_features += n_features
    
    X_poly = np.ones((n_samples, n_poly_features))
    col_idx = 1
    
    # Add features for each degree
    for d in range(1, degree + 1):
        for j in range(n_features):
            X_poly[:, col_idx] = X[:, j] ** d
            col_idx += 1
    
    return X_poly

def predict_with_polynomial_model(X, params):
    
    #Make predictions using the polynomial model
    
    #Parameters:
    #X : numpy array of feature values
    #params : dictionary of model parameters
    
    #Returns:
    #y_pred : predictions
    
    # Normalize features
    X_norm = (X - params['X_means']) / params['X_stds']
    
    # Generate polynomial features
    X_poly = generate_polynomial_features(X_norm, params['degree'])
    
    # Make predictions (normalized)
    y_pred_norm = X_poly @ params['coefficients']
    
    # Denormalize predictions
    y_pred = denormalize_predictions(y_pred_norm, params['y_mean'], params['y_std'])
    
    return y_pred

def train_ridge_model(X_train, y_train, alpha=1.0):
    
    #Train a ridge regression model
    
    #Parameters:
    #X_train : training features
    #y_train : training target values
    #alpha : regularization strength
    
    #Returns:
    #model : trained ridge regression model
    #X_mean : means used for normalization
    #X_std : standard deviations used for normalization
    #y_mean : mean of target values
    #y_std : standard deviation of target values
    
    # Normalize features and target
    X_norm, X_mean, X_std = normalize_data(X_train)
    y_norm, y_mean, y_std = normalize_data(y_train.reshape(-1, 1))
    y_norm = y_norm.flatten()
    
    # Train ridge model
    model = RidgeRegression(alpha=alpha, fit_intercept=True, normalize=False)
    model.fit(X_norm, y_norm)
    
    return model, X_mean, X_std, y_mean[0], y_std[0]

def predict_with_ridge_model(X, model, X_mean, X_std, y_mean, y_std):
    
    #Make predictions using the ridge model
    
    #Parameters:
    #X : features
    #model : trained ridge regression model
    #X_mean, X_std : normalization parameters for features
    #y_mean, y_std : normalization parameters for target
    
    #Returns:
    #y_pred : predictions
    
    #Normalize features
    X_norm = (X - X_mean) / X_std
    
    # Make predictions (normalized)
    y_pred_norm = model.predict(X_norm)
    
    # Denormalize predictions
    y_pred = y_pred_norm * y_std + y_mean
    
    return y_pred

def main():
    print("Stacking Polynomial and Ridge Regression Models for Housing Price Prediction")
    
    # Load training data
    train_path = "train_housingprices.csv"
    feature_cols = ['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']
    target_col = 'SalePrice'
    
    print("\nLoading training data...")
    train_data, X_train, y_train, _ = load_housing_data(train_path, feature_cols, target_col)
    print(f"Loaded {len(X_train)} training samples")
    
    # Load test data
    test_path = "test_housingprices.csv"
    print("\nLoading test data...")
    test_data, X_test, _, test_ids = load_housing_data(test_path, feature_cols, None)
    print(f"Loaded {len(X_test)} test samples")
    
    # Load polynomial model parameters
    poly_params_path = "polynomial_model_params.csv"
    print("\nLoading polynomial model parameters...")
    poly_params = load_polynomial_model_params(poly_params_path)
    print(f"Loaded polynomial model with degree {poly_params['degree']}")
    
    # Train ridge regression model
    print("\nTraining ridge regression model...")
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = 1.0  # Default
    best_score = -float('inf')
    
    # Simple cross-validation to find best alpha
    for alpha in alphas:
        # Use 80% of data for training, 20% for validation
        n_samples = len(X_train)
        n_train = int(0.8 * n_samples)
        
        # Random split
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        X_train_cv, y_train_cv = X_train[train_idx], y_train[train_idx]
        X_val_cv, y_val_cv = X_train[val_idx], y_train[val_idx]
        
        # Train model
        model, X_mean, X_std, y_mean, y_std = train_ridge_model(X_train_cv, y_train_cv, alpha)
        
        # Predict on validation set
        y_pred_cv = predict_with_ridge_model(X_val_cv, model, X_mean, X_std, y_mean, y_std)
        
        # Calculate R² score
        y_mean_val = np.mean(y_val_cv)
        ss_total = np.sum((y_val_cv - y_mean_val) ** 2)
        ss_residual = np.sum((y_val_cv - y_pred_cv) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        print(f"  Alpha = {alpha}: R² = {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} with R² = {best_score:.4f}")
    
    # Train final ridge model on all training data
    ridge_model, ridge_X_mean, ridge_X_std, ridge_y_mean, ridge_y_std = train_ridge_model(X_train, y_train, best_alpha)
    
    # Generate predictions for test data
    print("\nGenerating predictions...")
    
    # Polynomial model predictions
    poly_predictions = predict_with_polynomial_model(X_test, poly_params)
    
    # Ridge model predictions
    ridge_predictions = predict_with_ridge_model(X_test, ridge_model, ridge_X_mean, ridge_X_std, ridge_y_mean, ridge_y_std)
    
    # Stack predictions (simple average)
    stacked_predictions = (poly_predictions + ridge_predictions) / 2
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Id': test_ids,
        'Polynomial_SalePrice': poly_predictions,
        'Ridge_SalePrice': ridge_predictions,
        'SalePrice': stacked_predictions
    })
    
    # Save predictions to CSV
    output_path = "housing_stacked_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved stacked predictions to {output_path}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(output_df.head())
    
    # Plot predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(poly_predictions, ridge_predictions, alpha=0.5)
    plt.plot([min(poly_predictions), max(poly_predictions)], [min(poly_predictions), max(poly_predictions)], 'r--')
    plt.xlabel('Polynomial Regression Predictions')
    plt.ylabel('Ridge Regression Predictions')
    plt.title('Comparison of Model Predictions')
    plt.grid(True)
    plt.savefig('model_comparison.png')
    
    # Plot feature importance for ridge model
    feature_importance = np.abs(ridge_model.coef_)
    plt.figure(figsize=(10, 6))
    plt.bar(feature_cols, feature_importance)
    plt.xlabel('Features')
    plt.ylabel('Absolute Coefficient Value')
    plt.title('Ridge Model Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ridge_feature_importance.png')
    
    print("\nStacking complete! Predictions saved to housing_stacked_predictions.csv")

if __name__ == "__main__":
    main()
