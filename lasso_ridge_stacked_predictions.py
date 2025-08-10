import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def load_data(train_path, test_path, feature_cols=None):
    """
    Load housing data from CSV files
    
    Parameters:
    train_path : path to the training CSV file
    test_path : path to the test CSV file
    feature_cols : list of feature column names (None to use all numeric features)
    
    Returns:
    X_train : training feature matrix
    y_train : training target values
    X_test : test feature matrix
    test_ids : test IDs
    feature_names : list of feature names
    """
    # Load training data
    train_data = pd.read_csv(train_path)
    
    # Load test data
    test_data = pd.read_csv(test_path)
    
    # Extract test IDs
    test_ids = test_data['Id'].values
    
    # If no specific features are provided, use all numeric columns except Id and SalePrice
    if feature_cols is None:
        numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]
    
    # Check if all feature columns exist in both datasets
    for col in feature_cols:
        if col not in train_data.columns or col not in test_data.columns:
            raise ValueError(f"Feature column '{col}' not found in one of the datasets")
    
    # Extract features
    X_train = train_data[feature_cols].values
    X_test = test_data[feature_cols].values
    
    # Extract target
    y_train = train_data['SalePrice'].values
    
    return X_train, y_train, X_test, test_ids, feature_cols

def preprocess_data(X_train, X_test):
    """
    Preprocess data: handle missing values and standardize
    
    Parameters:
    X_train : training feature matrix
    X_test : test feature matrix
    
    Returns:
    X_train_scaled : preprocessed training features
    X_test_scaled : preprocessed test features
    scaler : fitted StandardScaler
    """
    # Handle missing values by replacing with mean from training data
    for col in range(X_train.shape[1]):
        train_mask = ~np.isnan(X_train[:, col])
        # Calculate mean from training data regardless of whether there are missing values
        col_mean = np.mean(X_train[train_mask, col])
        
        # Replace missing values in training data if any
        if not all(train_mask):
            X_train[~train_mask, col] = col_mean
            
        # Replace missing values in test data if any
        test_mask = ~np.isnan(X_test[:, col])
        if not all(test_mask):
            X_test[~test_mask, col] = col_mean
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def find_best_alpha(X, y, model_type='ridge', cv=5, alphas=None):
    """
    Find the best alpha parameter using cross-validation
    
    Parameters:
    X : feature matrix
    y : target values
    model_type : 'ridge' or 'lasso'
    cv : number of cross-validation folds
    alphas : list of alpha values to try
    
    Returns:
    best_alpha : best alpha value
    """
    if alphas is None:
        if model_type == 'ridge':
            alphas = np.logspace(-4, 4, 100)
        else:  # lasso
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
            if model_type == 'ridge':
                model = Ridge(alpha=alpha, fit_intercept=True, max_iter=10000)
            else:  # lasso
                model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
                
            model.fit(X_train, y_train)
            
            # Calculate validation MSE
            y_val_pred = model.predict(X_val)
            val_mse[fold, i] = np.mean((y_val - y_val_pred) ** 2)
    
    # Average validation MSE across folds
    avg_val_mse = np.mean(val_mse, axis=0)
    
    # Find best alpha
    best_alpha_idx = np.argmin(avg_val_mse)
    best_alpha = alphas[best_alpha_idx]
    
    return best_alpha

def train_models(X_train, y_train):
    """
    Train ridge and lasso models with optimal alpha values
    
    Parameters:
    X_train : training feature matrix
    y_train : training target values
    
    Returns:
    ridge_model : trained ridge model
    lasso_model : trained lasso model
    """
    print("Finding best alpha for Ridge regression...")
    ridge_alpha = find_best_alpha(X_train, y_train, model_type='ridge')
    print(f"Best Ridge alpha: {ridge_alpha:.6f}")
    
    print("Finding best alpha for Lasso regression...")
    lasso_alpha = find_best_alpha(X_train, y_train, model_type='lasso')
    print(f"Best Lasso alpha: {lasso_alpha:.6f}")
    
    # Train final models with best alphas
    ridge_model = Ridge(alpha=ridge_alpha, fit_intercept=True, max_iter=10000)
    ridge_model.fit(X_train, y_train)
    
    lasso_model = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    
    return ridge_model, lasso_model

def generate_predictions(X_test, ridge_model, lasso_model):
    """
    Generate stacked predictions from ridge and lasso models
    
    Parameters:
    X_test : test feature matrix
    ridge_model : trained ridge model
    lasso_model : trained lasso model
    
    Returns:
    stacked_predictions : stacked predictions
    """
    # Generate predictions
    ridge_predictions = ridge_model.predict(X_test)
    lasso_predictions = lasso_model.predict(X_test)
    
    # Stack predictions (simple average)
    stacked_predictions = (ridge_predictions + lasso_predictions) / 2
    
    return stacked_predictions, ridge_predictions, lasso_predictions

def main():
    print("Stacking Lasso and Ridge Regression Models for Housing Price Prediction")
    
    # Load data
    train_path = "train_housingprices.csv"
    test_path = "test_housingprices.csv"
    
    # Use the same features as in the original housing models
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
    
    print("\nLoading data...")
    X_train, y_train, X_test, test_ids, feature_names = load_data(train_path, test_path, feature_cols)
    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples with {X_train.shape[1]} features")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # Train models
    print("\nTraining models...")
    ridge_model, lasso_model = train_models(X_train_scaled, y_train)
    
    # Generate predictions
    print("\nGenerating predictions...")
    stacked_predictions, ridge_predictions, lasso_predictions = generate_predictions(X_test_scaled, ridge_model, lasso_model)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Id': test_ids,
        'Ridge_SalePrice': ridge_predictions,
        'Lasso_SalePrice': lasso_predictions,
        'SalePrice': stacked_predictions
    })
    
    # Save full predictions to CSV
    full_output_path = "housing_lasso_ridge_stacked_predictions.csv"
    output_df.to_csv(full_output_path, index=False)
    print(f"Saved full predictions to {full_output_path}")
    
    # Save final predictions in required format
    final_output_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': stacked_predictions
    })
    
    final_output_path = "housing_final_predictions.csv"
    final_output_df.to_csv(final_output_path, index=False)
    print(f"Saved final predictions to {final_output_path}")
    
    # Print top 10 most important features for each model
    print("\nTop 10 most important features for Ridge model:")
    ridge_coef_importance = sorted(zip(feature_names, ridge_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in ridge_coef_importance[:10]:
        print(f"{feature:<15}: {coef:.6f}")
    
    print("\nTop 10 most important features for Lasso model:")
    lasso_coef_importance = sorted(zip(feature_names, lasso_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in lasso_coef_importance[:10]:
        print(f"{feature:<15}: {coef:.6f}")

if __name__ == "__main__":
    main()
