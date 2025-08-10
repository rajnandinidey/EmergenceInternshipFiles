import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from lasso_regression import LassoRegression
from polynomial_regression import PolynomialRegression

def load_housing_data(file_path, feature_cols=None):
    """
    Load housing data from CSV file
    
    Parameters:
    file_path : path to the CSV file
    feature_cols : list of feature column names (if None, use default features)
    
    Returns:
    data : pandas DataFrame containing the data
    X : numpy array of features
    y : numpy array of target values (None for test data without target)
    ids : list of ID values
    """
    # Default features if none provided
    if feature_cols is None:
        feature_cols = [
            'GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual',
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
            'GarageCars'
        ]
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Extract IDs
    ids = data['Id'].values
    
    # Handle missing values in features
    for col in feature_cols:
        if col in data.columns:
            # Replace missing values with mean
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mean())
    
    # Extract features
    X = data[feature_cols].values
    
    # Extract target if available
    y = None
    if 'SalePrice' in data.columns:
        y = data['SalePrice'].values
    
    return data, X, y, ids

def normalize_data(X, X_mean=None, X_std=None):
    """
    Normalize features to prevent numerical issues
    
    Parameters:
    X : list or numpy array of feature values
    X_mean : means for normalization (calculated if None)
    X_std : standard deviations for normalization (calculated if None)
    
    Returns:
    X_norm : normalized features
    X_mean : means used for normalization
    X_std : standard deviations used for normalization
    """
    # Convert to numpy for calculations
    X_np = np.array(X)
    
    # Check if X is a scalar or 1D array
    is_1d = X_np.ndim == 1
    
    if is_1d:
        if X_mean is None:
            X_mean = np.mean(X_np)
        if X_std is None:
            X_std = np.std(X_np)
            if X_std < 1e-10:
                X_std = 1.0
        
        X_norm = (X_np - X_mean) / X_std
        X_norm = X_norm.tolist()
    else:
        # Handle feature matrix
        if X_mean is None:
            X_mean = np.mean(X_np, axis=0)
        if X_std is None:
            X_std = np.std(X_np, axis=0)
            # Prevent division by zero
            X_std[X_std < 1e-10] = 1.0
        
        X_norm = (X_np - X_mean) / X_std
        X_norm = X_norm.tolist()
    
    return X_norm, X_mean, X_std

def denormalize_predictions(y_pred_norm, y_mean, y_std):
    """
    Denormalize predictions back to original scale
    
    Parameters:
    y_pred_norm : normalized predictions (list or numpy array)
    y_mean : mean used for normalization
    y_std : standard deviation used for normalization
    
    Returns:
    y_pred : denormalized predictions
    """
    # Convert to numpy array for calculations
    y_pred_norm_np = np.array(y_pred_norm)
    
    # Denormalize
    y_pred = y_pred_norm_np * y_std + y_mean
    
    return y_pred

def train_lasso_model(X_train, y_train, alphas=None):
    """
    Train a Lasso regression model with cross-validation for alpha selection
    
    Parameters:
    X_train : training feature matrix
    y_train : training target values
    alphas : list of alpha values to try
    
    Returns:
    best_model : trained Lasso model with best alpha
    best_alpha : best alpha value
    """
    if alphas is None:
        alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Split data for cross-validation
    n_samples = len(X_train)
    n_val = int(0.2 * n_samples)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    X_train_cv = [X_train[i] for i in train_idx]
    y_train_cv = [y_train[i] for i in train_idx]
    X_val = [X_train[i] for i in val_idx]
    y_val = [y_train[i] for i in val_idx]
    
    # Train models with different alphas
    models = []
    mse_values = []
    
    print("Training Lasso models with different alphas...")
    for alpha in alphas:
        model = LassoRegression(alpha=alpha, max_iter=10000)
        model.fit(X_train_cv, y_train_cv)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        mse = np.mean((np.array(y_val) - np.array(y_val_pred)) ** 2)
        
        models.append(model)
        mse_values.append(mse)
        print(f"  Alpha {alpha:.6f} - Validation MSE: {mse:.6f}")
    
    # Find best model based on validation MSE
    best_idx = np.argmin(mse_values)
    best_alpha = alphas[best_idx]
    
    # Train final model on all data
    best_model = LassoRegression(alpha=best_alpha, max_iter=10000)
    best_model.fit(X_train, y_train)
    
    print(f"Best Lasso alpha: {best_alpha:.6f}")
    return best_model, best_alpha

def train_polynomial_model(X_train, y_train, degree=2):
    """
    Train a polynomial regression model
    
    Parameters:
    X_train : training feature matrix
    y_train : training target values
    degree : polynomial degree
    
    Returns:
    model : trained polynomial model
    """
    print(f"Training Polynomial model with degree {degree}...")
    model = PolynomialRegression(degree=degree)
    model.fit(X_train, y_train)
    return model

def main():
    print("Stacking Lasso and Polynomial Regression Models for Housing Price Prediction")
    
    # Define features to use
    feature_cols = [
        'GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual',
        'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
        'GarageCars'
    ]
    
    # Load data
    train_path = "train_housingprices.csv"
    test_path = "test_housingprices.csv"
    
    print("\nLoading training data...")
    train_data, X_train, y_train, _ = load_housing_data(train_path, feature_cols)
    print(f"Loaded {len(X_train)} training samples with {len(feature_cols)} features")
    
    print("\nLoading test data...")
    test_data, X_test, _, test_ids = load_housing_data(test_path, feature_cols)
    print(f"Loaded {len(X_test)} test samples")
    
    # Ensure data is in list format for compatibility with polynomial regression
    X_train = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
    y_train = y_train.tolist() if isinstance(y_train, np.ndarray) else y_train
    X_test = X_test.tolist() if isinstance(X_test, np.ndarray) else X_test
    
    # Normalize data
    print("\nNormalizing data...")
    X_train_norm, X_mean, X_std = normalize_data(X_train)
    y_train_norm, y_mean, y_std = normalize_data(y_train)
    X_test_norm, _, _ = normalize_data(X_test, X_mean, X_std)
    
    # Train Lasso model
    print("\nTraining Lasso model...")
    lasso_model, best_alpha = train_lasso_model(X_train_norm, y_train_norm)
    
    # Train Polynomial model
    print("\nTraining Polynomial model...")
    poly_model = train_polynomial_model(X_train_norm, y_train_norm, degree=2)
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    # Lasso predictions
    lasso_pred_norm = lasso_model.predict(X_test_norm)
    lasso_pred = denormalize_predictions(lasso_pred_norm, y_mean, y_std)
    
    # Polynomial predictions
    poly_pred_norm = poly_model.predict(X_test_norm)
    poly_pred = denormalize_predictions(poly_pred_norm, y_mean, y_std)
    
    # Convert predictions to numpy arrays for calculations
    lasso_pred_np = np.array(lasso_pred)
    poly_pred_np = np.array(poly_pred)
    
    # Stack predictions (simple average)
    stacked_pred = (lasso_pred_np + poly_pred_np) / 2
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Id': test_ids,
        'Lasso_SalePrice': lasso_pred,
        'Polynomial_SalePrice': poly_pred,
        'SalePrice': stacked_pred
    })
    
    # Save full predictions to CSV
    full_output_path = "housing_lasso_polynomial_stacked_predictions.csv"
    output_df.to_csv(full_output_path, index=False)
    print(f"Saved full predictions to {full_output_path}")
    
    # Save final predictions in required format
    final_output_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': stacked_pred
    })
    
    final_output_path = "housing_lasso_polynomial_final_predictions.csv"
    final_output_df.to_csv(final_output_path, index=False)
    print(f"Saved final predictions to {final_output_path}")
    
    # Print feature importance for Lasso model
    print("\nLasso model feature importance:")
    lasso_coef = list(zip(feature_cols, lasso_model.coef_))
    lasso_coef.sort(key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in lasso_coef:
        print(f"{feature:<15}: {coef:.6f}")

if __name__ == "__main__":
    main()
