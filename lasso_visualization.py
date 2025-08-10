import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

def load_housing_data(file_path, feature_cols=None):
    """
    Load housing data from CSV file
    
    Parameters:
    file_path : path to the CSV file
    feature_cols : list of feature column names (None to use all numeric features)
    
    Returns:
    X : feature matrix
    y : target values
    feature_names : list of feature names
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # If no specific features are provided, use all numeric columns except Id and SalePrice
    if feature_cols is None:
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]
    
    # Check if all feature columns exist
    for col in feature_cols:
        if col not in data.columns:
            raise ValueError(f"Feature column '{col}' not found in the data")
    
    # Extract features and target
    X = data[feature_cols].values
    
    # Extract target if available
    y = None
    if 'SalePrice' in data.columns:
        y = data['SalePrice'].values
    
    return X, y, feature_cols

def plot_coefficient_path(X, y, feature_names, alphas=None, n_alphas=100):
    """
    Generate coefficient path plot for Lasso regression
    
    Parameters:
    X : feature matrix
    y : target values
    feature_names : list of feature names
    alphas : list of alpha values (None to generate automatically)
    n_alphas : number of alpha values to generate if alphas is None
    """
    # Generate alphas if not provided
    if alphas is None:
        alphas = np.logspace(-4, 1, n_alphas)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize coefficients array
    coefs = np.zeros((len(alphas), X.shape[1]))
    
    # Compute coefficients for each alpha
    for i, alpha in enumerate(alphas):
        lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        lasso.fit(X_scaled, y)
        coefs[i] = lasso.coef_
    
    # Plot coefficient paths
    plt.figure(figsize=(12, 8))
    
    # Plot only top 15 features with highest coefficient variance
    coef_variance = np.var(coefs, axis=0)
    top_features = np.argsort(-coef_variance)[:15]
    
    for i in top_features:
        plt.semilogx(alphas, coefs[:, i], label=feature_names[i])
    
    plt.xlabel('Alpha (regularization parameter)')
    plt.ylabel('Coefficient value')
    plt.title('Lasso Coefficient Path')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-")
    plt.axhline(y=0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig('lasso_coefficient_path.png')
    plt.close()

def plot_cv_mse_curve(X, y, alphas=None, n_alphas=100, cv=5):
    """
    Generate cross-validation MSE curve for Lasso regression
    
    Parameters:
    X : feature matrix
    y : target values
    alphas : list of alpha values (None to generate automatically)
    n_alphas : number of alpha values to generate if alphas is None
    cv : number of cross-validation folds
    """
    # Generate alphas if not provided
    if alphas is None:
        alphas = np.logspace(-4, 1, n_alphas)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize arrays for MSE values
    train_mse = np.zeros(len(alphas))
    val_mse = np.zeros((cv, len(alphas)))
    
    # Perform cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for i, alpha in enumerate(alphas):
            # Train model
            lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
            lasso.fit(X_train, y_train)
            
            # Calculate validation MSE
            y_val_pred = lasso.predict(X_val)
            val_mse[fold, i] = mean_squared_error(y_val, y_val_pred)
            
            # Calculate training MSE for the first fold only
            if fold == 0:
                y_train_pred = lasso.predict(X_train)
                train_mse[i] = mean_squared_error(y_train, y_train_pred)
    
    # Average validation MSE across folds
    avg_val_mse = np.mean(val_mse, axis=0)
    
    # Find best alpha
    best_alpha_idx = np.argmin(avg_val_mse)
    best_alpha = alphas[best_alpha_idx]
    
    # Plot MSE curves
    plt.figure(figsize=(12, 8))
    plt.semilogx(alphas, train_mse, label='Training MSE')
    plt.semilogx(alphas, avg_val_mse, label='Validation MSE')
    plt.xlabel('Alpha (regularization parameter)')
    plt.ylabel('Mean Squared Error')
    plt.title('Lasso Cross-Validation MSE Curve')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig('lasso_cv_mse_curve.png')
    plt.close()
    
    return best_alpha, avg_val_mse[best_alpha_idx]

def main():
    print("Lasso Regression Visualization")
    
    # Load training data
    train_path = "train_housingprices.csv"
    
    # Use the same features as in the original housing_lasso_regression.py
    feature_cols = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
        'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal', 'YrSold'
    ]
    
    print("\nLoading training data...")
    X, y, feature_names = load_housing_data(train_path, feature_cols)
    
    # Handle missing values by replacing with mean
    for col in range(X.shape[1]):
        mask = ~np.isnan(X[:, col])
        if not all(mask):
            X[~mask, col] = np.mean(X[mask, col])
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Define a wider range of alpha values
    alphas = np.logspace(-4, 2, 100)
    
    # Plot coefficient path
    print("\nGenerating coefficient path plot...")
    plot_coefficient_path(X, y, feature_names, alphas)
    print("Saved coefficient path plot to 'lasso_coefficient_path.png'")
    
    # Plot CV MSE curve
    print("\nGenerating cross-validation MSE curve...")
    best_alpha, best_mse = plot_cv_mse_curve(X, y, alphas)
    print(f"Best alpha: {best_alpha:.6f} with MSE: {best_mse:.2f}")
    print("Saved CV MSE curve to 'lasso_cv_mse_curve.png'")
    
    # Train final model with best alpha
    print("\nTraining final model with best alpha...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = Lasso(alpha=best_alpha, fit_intercept=True, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Print top 10 most important features
    coef_importance = sorted(zip(feature_names, lasso.coef_), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 10 most important features:")
    for feature, coef in coef_importance[:10]:
        print(f"{feature:<15}: {coef:.6f}")
    
    # Count non-zero coefficients
    non_zero = sum(1 for coef in lasso.coef_ if abs(coef) > 1e-8)
    print(f"\nNon-zero coefficients: {non_zero}/{len(lasso.coef_)}")

if __name__ == "__main__":
    main()
