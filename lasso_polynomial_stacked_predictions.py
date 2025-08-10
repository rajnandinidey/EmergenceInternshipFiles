import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from enhanced_polynomial_regression import EnhancedPolynomialRegression

def load_data(train_path, test_path):
    """
    Load housing data from CSV files
    
    Parameters:
    train_path : path to the training CSV file
    test_path : path to the test CSV file
    
    Returns:
    X_train : training feature matrix
    y_train : training target values
    X_test : test feature matrix
    test_ids : test IDs
    feature_names : list of feature names
    """
    # Define features to use
    feature_cols = ['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']
    
    # Load training data
    train_data = pd.read_csv(train_path)
    X_train = train_data[feature_cols].values
    y_train = train_data['SalePrice'].values
    
    # Load test data
    test_data = pd.read_csv(test_path)
    test_ids = test_data['Id'].values
    X_test = test_data[feature_cols].values
    
    return X_train, y_train, X_test, test_ids, feature_cols

def normalize_data(X, X_mean=None, X_std=None):
    """
    Normalize features to prevent numerical issues
    
    Parameters:
    X : numpy array of feature values or scalar
    X_mean : means for normalization (calculated if None)
    X_std : standard deviations for normalization (calculated if None)
    
    Returns:
    X_norm : normalized features
    X_mean : means used for normalization
    X_std : standard deviations used for normalization
    """
    # Check if X is a scalar (for target values)
    if np.isscalar(X) or (isinstance(X, np.ndarray) and X.ndim == 1 and len(X.shape) == 1):
        if X_mean is None:
            X_mean = np.mean(X)
        if X_std is None:
            X_std = np.std(X)
            if X_std < 1e-10:
                X_std = 1.0
        
        X_norm = (X - X_mean) / X_std
        return X_norm, X_mean, X_std
    else:
        # Handle feature matrix
        if X_mean is None:
            X_mean = np.mean(X, axis=0)
        if X_std is None:
            X_std = np.std(X, axis=0)
            # Prevent division by zero
            X_std[X_std < 1e-10] = 1.0
        
        X_norm = (X - X_mean) / X_std
        return X_norm, X_mean, X_std

def denormalize_predictions(y_pred_norm, y_mean, y_std):
    """
    Denormalize predictions back to original scale
    
    Parameters:
    y_pred_norm : normalized predictions
    y_mean : mean used for normalization
    y_std : standard deviation used for normalization
    
    Returns:
    y_pred : denormalized predictions
    """
    return y_pred_norm * y_std + y_mean

def find_best_alpha_lasso(X, y, cv=5, alphas=None):
    """
    Find the best alpha parameter for Lasso using cross-validation
    
    Parameters:
    X : feature matrix
    y : target values
    cv : number of cross-validation folds
    alphas : list of alpha values to try
    
    Returns:
    best_alpha : best alpha value
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 100)
    
    # Initialize arrays for MSE values
    val_mse = np.zeros((cv, len(alphas)))
    
    # Perform cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for i, alpha in enumerate(alphas):
            # Train model
            lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
            lasso.fit(X_train, y_train)
            
            # Calculate validation MSE
            y_val_pred = lasso.predict(X_val)
            val_mse[fold, i] = np.mean((y_val - y_val_pred) ** 2)
    
    # Average validation MSE across folds
    avg_val_mse = np.mean(val_mse, axis=0)
    
    # Find best alpha
    best_alpha_idx = np.argmin(avg_val_mse)
    best_alpha = alphas[best_alpha_idx]
    
    return best_alpha

def train_lasso_model(X_train, y_train, alpha):
    """
    Train a Lasso regression model
    
    Parameters:
    X_train : training feature matrix
    y_train : training target values
    alpha : regularization parameter
    
    Returns:
    model : trained Lasso model
    """
    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    lasso.fit(X_train, y_train)
    return lasso

def train_polynomial_model(X_train, y_train, degree=2, include_interactions=True):
    """
    Train a polynomial regression model with interactions
    
    Parameters:
    X_train : training feature matrix
    y_train : training target values
    degree : polynomial degree
    include_interactions : whether to include interaction terms
    
    Returns:
    model : trained polynomial model
    """
    model = EnhancedPolynomialRegression(degree=degree, include_interactions=include_interactions)
    model.fit(X_train, y_train)
    return model

def main():
    print("Stacking Lasso and Polynomial Regression Models for Housing Price Prediction")
    
    # Load data
    train_path = "train_housingprices.csv"
    test_path = "test_housingprices.csv"
    
    print("\nLoading data...")
    X_train, y_train, X_test, test_ids, feature_names = load_data(train_path, test_path)
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples with {X_train.shape[1]} features")
    
    # Normalize data
    print("\nNormalizing data...")
    X_train_norm, X_mean, X_std = normalize_data(X_train)
    y_train_norm, y_mean, y_std = normalize_data(y_train)
    X_test_norm, _, _ = normalize_data(X_test, X_mean, X_std)
    
    # Train Lasso model
    print("\nTraining Lasso model...")
    best_alpha = find_best_alpha_lasso(X_train_norm, y_train_norm)
    print(f"Best Lasso alpha: {best_alpha:.6f}")
    lasso_model = train_lasso_model(X_train_norm, y_train_norm, best_alpha)
    
    # Train Polynomial model
    print("\nTraining Polynomial model with interactions...")
    poly_model = train_polynomial_model(X_train_norm, y_train_norm, degree=2, include_interactions=True)
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    # Lasso predictions
    lasso_pred_norm = lasso_model.predict(X_test_norm)
    lasso_pred = denormalize_predictions(lasso_pred_norm, y_mean, y_std)
    
    # Polynomial predictions
    poly_pred_norm = poly_model.predict(X_test_norm)
    poly_pred = denormalize_predictions(poly_pred_norm, y_mean, y_std)
    
    # Stack predictions (simple average)
    stacked_pred = (lasso_pred + poly_pred) / 2
    
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
    lasso_coef = list(zip(feature_names, lasso_model.coef_))
    lasso_coef.sort(key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in lasso_coef:
        print(f"{feature:<15}: {coef:.6f}")
    
    # Print feature importance for Polynomial model
    print("\nPolynomial model feature importance:")
    poly_coef = list(zip(poly_model.feature_names, poly_model.coefficients))
    poly_coef.sort(key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in poly_coef[:10]:  # Show top 10
        print(f"{feature:<15}: {coef:.6f}")

if __name__ == "__main__":
    main()
