import numpy as np
import matplotlib.pyplot as plt
import csv
from lasso_regression import LassoRegression

def load_housing_data(file_path):
    """Load housing data from CSV file"""
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
            
            data.append(processed_row)
    
    return headers, data

def normalize_data(X):
    """Normalize features to prevent numerical issues"""
    X_mean = sum(X) / len(X)
    X_std = (sum((x - X_mean) ** 2 for x in X) / len(X)) ** 0.5
    X_std = max(X_std, 1e-8)  # Prevent division by zero
    
    X_norm = [(x - X_mean) / X_std for x in X]
    
    return X_norm, X_mean, X_std

def detect_outliers_with_lasso(file_path):
    """
    Detect potential outliers in housing data using Lasso regression
    by identifying features that get coefficients of 0
    """
    # Load housing data
    print("Loading housing data...")
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
    
    # Try different alpha values to see which features get zeroed out
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    
    # Store results for each alpha
    results = []
    
    print("\n===== ANALYZING FEATURES WITH LASSO REGRESSION =====")
    for alpha in alphas:
        print(f"Training Lasso regression model with alpha={alpha}...")
        
        # Create and fit the model
        model = LassoRegression(alpha=alpha, max_iter=5000)
        model.fit(X_norm, y_norm)
        
        # Get coefficients
        coefficients = model.coef_
        
        # Count non-zero coefficients
        non_zero = sum(1 for coef in coefficients if abs(coef) > 1e-8)
        zero_coefs = len(coefficients) - non_zero
        
        print(f"  Alpha {alpha} - Features zeroed out: {zero_coefs}/{len(coefficients)}")
        
        # Identify which features were zeroed out
        zeroed_features = []
        for i, coef in enumerate(coefficients):
            if abs(coef) <= 1e-8:
                zeroed_features.append(numeric_features[i])
        
        results.append({
            'alpha': alpha,
            'non_zero_count': non_zero,
            'zero_count': zero_coefs,
            'zeroed_features': zeroed_features,
            'coefficients': coefficients
        })
    
    # Print detailed results
    print("\n===== FEATURES ZEROED OUT BY ALPHA VALUE =====")
    for result in results:
        alpha = result['alpha']
        zeroed_features = result['zeroed_features']
        
        print(f"\nAlpha = {alpha}:")
        if zeroed_features:
            for feature in zeroed_features:
                print(f"  - {feature}")
        else:
            print("  No features were zeroed out")
    
    # Plot number of zeroed features vs alpha
    plt.figure(figsize=(10, 6))
    alphas_log = alphas  # For log scale plotting
    zero_counts = [result['zero_count'] for result in results]
    
    plt.plot(alphas_log, zero_counts, 'bo-', markersize=8)
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Number of Features Zeroed Out')
    plt.title('Feature Selection with Lasso Regression')
    plt.grid(True)
    plt.savefig('lasso_feature_zeroing.png')
    
    # Plot coefficient values for each feature across different alphas
    plt.figure(figsize=(14, 8))
    
    # Get coefficient values for each feature across alphas
    for i, feature in enumerate(numeric_features):
        feature_coefs = [result['coefficients'][i] for result in results]
        plt.plot(alphas_log, feature_coefs, 'o-', label=feature if i < 10 else None)
    
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Coefficients vs Regularization Strength')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    plt.tight_layout()
    plt.savefig('lasso_coefficients_by_alpha.png')
    
    # For the highest alpha value, show feature importance of non-zero features
    highest_alpha_result = results[-1]  # Last result has highest alpha
    
    # Get feature importance
    feature_importance = [(numeric_features[i], highest_alpha_result['coefficients'][i]) 
                         for i in range(len(numeric_features))]
    
    # Filter non-zero features and sort by absolute coefficient value
    non_zero_features = [(feature, coef) for feature, coef in feature_importance 
                        if abs(coef) > 1e-8]
    non_zero_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n===== MOST IMPORTANT FEATURES (HIGHEST ALPHA) =====")
    for feature, coef in non_zero_features:
        print(f"{feature:<15}\t{coef:.6f}")
    
    # Plot feature importance for highest alpha
    plt.figure(figsize=(12, 8))
    features = [f[0] for f in non_zero_features]
    coefficients = [f[1] for f in non_zero_features]
    
    y_pos = np.arange(len(features))
    plt.barh(y_pos, coefficients, align='center')
    plt.yticks(y_pos, features)
    plt.xlabel('Coefficient Value')
    plt.title(f'Feature Importance (Alpha = {highest_alpha_result["alpha"]})')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('lasso_feature_importance.png')
    
    print("\n===== ANALYSIS COMPLETED =====")
    print("Results saved to lasso_feature_zeroing.png, lasso_coefficients_by_alpha.png, and lasso_feature_importance.png")
    
    return results

if __name__ == "__main__":
    file_path = "train_housingprices.csv"
    results = detect_outliers_with_lasso(file_path)
