import csv
import matplotlib.pyplot as plt
import numpy as np
from lasso_regression import LassoRegression

def load_housing_data(file_path):
    #Load housing data from CSV file
    
    #Parameters:
    #file_path : path to the CSV file
    
    #Returns:
    #headers : list of column names
    #data : list of lists containing the data
    
    headers = []
    data = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get header row
        
        for row in csv_reader:
            # Process the row
            processed_row = []
            for value in row:
                if value == 'NA' or value == '':
                    processed_row.append(None)
                else:
                    try:
                        processed_row.append(float(value))
                    except ValueError:
                        processed_row.append(value)  # Keep string values as is
            
            # Only add rows with all numeric values for selected features
            data.append(processed_row)
    
    return headers, data

def normalize_data(X):
    #Normalize features to prevent numerical issues
    
    #Parameters:
    #X : list of feature values
    
    #Returns:
    #X_norm : normalized features
    #X_mean, X_std : mean and standard deviation of X
    
    X_mean = sum(X) / len(X)
    X_std = (sum((x - X_mean) ** 2 for x in X) / len(X)) ** 0.5
    X_std = max(X_std, 1e-8)  # Prevent division by zero
    
    X_norm = [(x - X_mean) / X_std for x in X]
    
    return X_norm, X_mean, X_std

def denormalize_parameters(coefficients, intercept, X_means, X_stds, y_mean, y_std):
    #Denormalize model parameters back to original scale
    
    #Parameters:
    #coefficients : list of normalized coefficients
    #intercept : normalized intercept
    #X_means : list of feature means
    #X_stds : list of feature standard deviations
    #y_mean : target mean
    #y_std : target standard deviation
    
    #Returns:
    #denorm_coefficients : denormalized coefficients
    #denorm_intercept : denormalized intercept
    
    denorm_coefficients = []
    for i, coef in enumerate(coefficients):
        denorm_coefficients.append(coef * y_std / X_stds[i])
    
    denorm_intercept = y_mean
    for i, coef in enumerate(coefficients):
        denorm_intercept -= denorm_coefficients[i] * X_means[i]
    
    return denorm_coefficients, denorm_intercept

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error between true and predicted values"""
    return sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)

def r_squared(y_true, y_pred):
    """Calculate R-squared (coefficient of determination)"""
    y_mean = sum(y_true) / len(y_true)
    ss_total = sum((y - y_mean) ** 2 for y in y_true)
    ss_residual = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    return 1 - (ss_residual / ss_total)

def main():
    # Load housing data
    print("Loading housing data...")
    file_path = "train_housingprices.csv"
    headers, data = load_housing_data(file_path)
    
    # Select numeric features to use (excluding Id and target)
    numeric_features = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
        'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal', 'YrSold'
    ]
    
    # Find indices of selected features
    feature_indices = []
    for feature in numeric_features:
        if feature in headers:
            feature_indices.append(headers.index(feature))
    
    # Find index of target (SalePrice)
    target_index = headers.index('SalePrice')
    
    # Extract valid rows (no None values in selected features or target)
    valid_rows = []
    for row in data:
        if (target_index < len(row) and row[target_index] is not None and
            all(i < len(row) and row[i] is not None and isinstance(row[i], (int, float)) 
                for i in feature_indices)):
            valid_rows.append(row)
    
    print(f"Loaded {len(valid_rows)} valid rows of data")
    
    # Extract features and target
    X = [[row[i] for i in feature_indices] for row in valid_rows]
    y = [row[target_index] for row in valid_rows]
    
    # Normalize data
    print("Normalizing data...")
    X_columns = list(zip(*X))  # Transpose to get columns
    X_norm_columns = []
    X_means = []
    X_stds = []
    
    for col in X_columns:
        col_norm, col_mean, col_std = normalize_data(col)
        X_norm_columns.append(col_norm)
        X_means.append(col_mean)
        X_stds.append(col_std)
    
    # Transpose back to rows
    X_norm = list(zip(*X_norm_columns))
    
    # Normalize target
    y_norm, y_mean, y_std = normalize_data(y)
    
    # Try different alpha values
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    models = []
    mse_values = []
    r_squared_values = []
    non_zero_coefs = []
    
    print("\n===== TRAINING MODELS =====")
    for alpha in alphas:
        print(f"Training Lasso regression model with alpha={alpha}...")
        
        # Create and fit the model
        model = LassoRegression(alpha=alpha, max_iter=5000)
        model.fit(X_norm, y_norm)
        
        # Make predictions (in normalized space)
        y_pred_norm = model.predict(X_norm)
        
        # Calculate error metrics
        mse = mean_squared_error(y_norm, y_pred_norm)
        r_squared_val = r_squared(y_norm, y_pred_norm)
        
        # Count non-zero coefficients
        non_zero = sum(1 for coef in model.coef_ if abs(coef) > 1e-8)
        
        print(f"  Alpha {alpha} - MSE: {mse:.6f}, R-squared: {r_squared_val:.6f}, Non-zero coefficients: {non_zero}/{len(model.coef_)}")
        
        models.append(model)
        mse_values.append(mse)
        r_squared_values.append(r_squared_val)
        non_zero_coefs.append(non_zero)
    
    # Find best model based on MSE
    best_index = mse_values.index(min(mse_values))
    best_alpha = alphas[best_index]
    best_model = models[best_index]
    
    print(f"\nBest model: Alpha={best_alpha}")
    print(f"MSE: {mse_values[best_index]:.6f}")
    print(f"R-squared: {r_squared_values[best_index]:.6f}")
    print(f"Non-zero coefficients: {non_zero_coefs[best_index]}/{len(best_model.coef_)}")
    
    # Denormalize coefficients for interpretation
    denorm_coefs, denorm_intercept = denormalize_parameters(
        best_model.coef_, best_model.intercept_, X_means, X_stds, y_mean, y_std
    )
    
    # Print feature importance
    print("\n===== FEATURE IMPORTANCE =====")
    feature_importance = [(numeric_features[i], denorm_coefs[i]) for i in range(len(numeric_features))]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Feature\t\tCoefficient")
    print("-" * 30)
    for feature, coef in feature_importance:
        if abs(coef) > 1e-8:  # Only show non-zero coefficients
            print(f"{feature:<15}\t{coef:.4f}")
    
    # Make predictions on training data
    y_pred = [pred * y_std + y_mean for pred in best_model.predict(X_norm)]
    
    # Print some sample predictions
    print("\n===== SAMPLE PREDICTIONS =====")
    sample_indices = np.random.choice(len(X), 5, replace=False)
    for i in sample_indices:
        print(f"Actual SalePrice = ${y[i]:.2f}, Predicted SalePrice = ${y_pred[i]:.2f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.title('Actual vs Predicted Housing Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.savefig('housing_lasso_regression.png')
    
    # Plot coefficient values for different alphas
    plt.figure(figsize=(12, 8))
    for i, alpha in enumerate(alphas):
        non_zero = sum(1 for coef in models[i].coef_ if abs(coef) > 1e-8)
        plt.plot(alphas[i], non_zero, 'bo', markersize=10)
    
    plt.xscale('log')
    plt.title('Number of Non-Zero Coefficients vs Alpha')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Number of Non-Zero Coefficients')
    plt.grid(True)
    plt.savefig('lasso_feature_selection.png')
    
    print("\n===== TRAINING COMPLETED =====")
    print("Results saved to housing_lasso_regression.png and lasso_feature_selection.png")

if __name__ == "__main__":
    main()
