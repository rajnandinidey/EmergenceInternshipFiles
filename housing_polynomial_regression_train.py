import csv
import matplotlib.pyplot as plt
import numpy as np
from polynomial_regression import PolynomialRegression

def load_housing_data(file_path, feature_cols=['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual'], target_col='SalePrice'):
    """
    Load housing data from CSV file
    
    Parameters:
    file_path : path to the CSV file
    feature_cols : list of feature column names
    target_col : name of the target column
    
    Returns:
    headers : list of column names
    data : list of lists containing the data with valid feature and target values
    feature_names : list of feature names that were successfully loaded
    """
    headers = []
    data = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get header row
        
        # Find the indices of the feature and target columns
        try:
            feature_indices = [headers.index(col) for col in feature_cols]
            target_idx = headers.index(target_col)
            feature_names = feature_cols.copy()  # Store the feature names that were found
        except ValueError as e:
            print(f"Error: Could not find one of the columns in the CSV headers: {e}")
            return headers, [], []
        
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
            
            # Only add rows where all feature and target values are valid numbers
            if target_idx < len(processed_row) and processed_row[target_idx] is not None:
                valid_features = True
                for idx in feature_indices:
                    if idx >= len(processed_row) or processed_row[idx] is None:
                        valid_features = False
                        break
                if valid_features:
                    data.append(processed_row)
    
    print(f"Loaded {len(data)} rows with valid features and {target_col} values")
    return headers, data, feature_names

def normalize_data(X):
    """
    Normalize features to prevent numerical issues
    
    Parameters:
    X : list of feature values
    
    Returns:
    X_norm : normalized features
    X_mean, X_std : mean and standard deviation of X
    """
    X_mean = sum(X) / len(X)
    X_std = (sum((x - X_mean) ** 2 for x in X) / len(X)) ** 0.5
    X_std = max(X_std, 1e-8)  # Prevent division by zero
    
    X_norm = [(x - X_mean) / X_std for x in X]
    
    return X_norm, X_mean, X_std

def denormalize_predictions(y_pred_norm, y_mean, y_std):
    """
    Denormalize predictions back to original scale.
    
    Parameters:
    y_pred_norm (list): normalized predictions
    y_mean (float): mean of y
    y_std (float): standard deviation of y
    
    Returns:
    list: predictions in original scale
    """
    
    return [y * y_std + y_mean for y in y_pred_norm]

def calculate_r_squared(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).
    
    Parameters:
    y_true (list): list of true values
    y_pred (list): list of predicted values
    
    Returns:
    float: coefficient of determination
    """
    
    y_mean = sum(y_true) / len(y_true)
    ss_total = sum([(y - y_mean) ** 2 for y in y_true])
    ss_residual = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
    
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def analyze_feature_dominance(coefficients, feature_names, degree):
    """
    Analyze which features are dominant in the polynomial regression model
    by examining the coefficient magnitudes.
    
    Parameters:
    coefficients (list): List of coefficients from the polynomial regression model
    feature_names (list): List of feature names
    degree (int): Degree of the polynomial
    """
    # The first coefficient is the intercept
    print(f"Intercept: {coefficients[0]}")
    
    # For each feature, get its coefficients for different powers
    feature_importance = {}
    
    # For each feature, calculate the sum of absolute values of its coefficients
    for i, feature in enumerate(feature_names):
        # Linear term (power 1)
        linear_coef = abs(coefficients[i + 1])
        
        # Higher-order terms if degree > 1
        higher_order_coefs = []
        for power in range(2, degree + 1):
            # Calculate the index for this feature's power term
            # For degree=2: [intercept, x1, x2, x3, x4, x1^2, x2^2, x3^2, x4^2]
            # For degree=3: [intercept, x1, x2, x3, x4, x1^2, x2^2, x3^2, x4^2, x1^3, x2^3, x3^3, x4^3]
            idx = len(feature_names) * (power - 1) + i + 1
            if idx < len(coefficients):
                higher_order_coefs.append(abs(coefficients[idx]))
        
        # Calculate total importance as sum of absolute coefficient values
        total_importance = linear_coef + sum(higher_order_coefs)
        feature_importance[feature] = {
            'linear': linear_coef,
            'higher_order': higher_order_coefs,
            'total': total_importance
        }
    
    # Sort features by total importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['total'], reverse=True)
    
    # Print feature importance
    print("\nFeature importance based on coefficient magnitudes:")
    for feature, importance in sorted_features:
        print(f"\n{feature}:")
        print(f"  Linear term coefficient (absolute): {importance['linear']:.6f}")
        for i, coef in enumerate(importance['higher_order']):
            print(f"  Power {i+2} coefficient (absolute): {coef:.6f}")
        print(f"  Total importance: {importance['total']:.6f}")
    
    # Identify the most dominant feature
    most_dominant = sorted_features[0][0]
    print(f"\nMost dominant feature: {most_dominant} with total importance of {sorted_features[0][1]['total']:.6f}")
    
    # Identify the most dominant term (linear or higher order)
    for feature, importance in sorted_features:
        max_coef = max([importance['linear']] + importance['higher_order'])
        if max_coef == importance['linear']:
            term = "linear"
            power = 1
        else:
            power = importance['higher_order'].index(max_coef) + 2
            term = f"power {power}"
        
        print(f"{feature}: Most dominant term is {term} with coefficient magnitude {max_coef:.6f}")

def main():
    # Load housing data
    print("Loading housing data...")
    file_path = "train_housingprices.csv"
    # Select multiple features for the model
    feature_cols = ['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']
    target_col = 'SalePrice'
    headers, data, feature_names = load_housing_data(file_path, feature_cols, target_col)
    
    # Find the indices of the feature and target columns
    feature_indices = [headers.index(col) for col in feature_names]
    target_idx = headers.index(target_col)
    
    # Extract feature and target values
    X = [[row[idx] for idx in feature_indices] for row in data]
    y = [row[target_idx] for row in data]
    
    # Normalize data
    print("Normalizing data...")
    # For multiple features, normalize each feature separately
    X_norm = []
    X_means = []
    X_stds = []
    
    # Transpose X to get a list of feature values for each feature
    X_transposed = list(zip(*X))
    
    for feature_values in X_transposed:
        feature_norm, feature_mean, feature_std = normalize_data(feature_values)
        X_means.append(feature_mean)
        X_stds.append(feature_std)
    
    # Transpose back to get normalized values for each sample
    X_norm_transposed = []
    for i in range(len(X_transposed)):
        feature_values = X_transposed[i]
        feature_norm, feature_mean, feature_std = normalize_data(feature_values)
        X_norm_transposed.append(feature_norm)
    
    X_norm = list(zip(*X_norm_transposed))
    
    # Normalize target values
    y_norm, y_mean, y_std = normalize_data(y)
    
    # Try different polynomial degrees
    degrees = [1, 2, 3]
    models = []
    mse_values = []
    r_squared_values = []
    
    print("\n===== TRAINING MODELS =====")
    for degree in degrees:
        print(f"Training polynomial regression model with degree {degree}...")
        
        # Create and fit the model
        model = PolynomialRegression(degree=degree)
        model.fit(X_norm, y_norm)
        
        # Make predictions (in normalized space)
        y_pred_norm = model.predict(X_norm)
        
        # Denormalize predictions
        y_pred = denormalize_predictions(y_pred_norm, y_mean, y_std)
        
        # Calculate error metrics
        mse = model.mean_squared_error(y_norm, y_pred_norm)
        r_squared = calculate_r_squared(y, y_pred)
        
        print(f"  Degree {degree} - MSE (normalized): {mse:.6f}, R-squared: {r_squared:.6f}")
        
        models.append(model)
        mse_values.append(mse)
        r_squared_values.append(r_squared)
    
    # Find best model based on MSE
    best_index = mse_values.index(min(mse_values))
    best_degree = degrees[best_index]
    best_model = models[best_index]
    
    print(f"\nBest model: Polynomial degree {best_degree}")
    print(f"Coefficients: {best_model.coefficients}")
    print(f"R-squared: {r_squared_values[best_index]:.6f}")
    
    # Analyze feature dominance by examining coefficient magnitudes
    print("\n===== FEATURE DOMINANCE ANALYSIS =====")
    analyze_feature_dominance(best_model.coefficients, feature_names, best_degree)
    
    # Make predictions with the best model (in normalized space)
    y_pred_norm = best_model.predict(X_norm)
    
    # Denormalize predictions
    y_pred = denormalize_predictions(y_pred_norm, y_mean, y_std)
    
    # Print some sample predictions
    print("\n===== SAMPLE PREDICTIONS =====")
    sample_indices = np.random.choice(len(X), 5, replace=False)
    for i in sample_indices:
        features_str = ", ".join([f"{feature_names[j]} = {X[i][j]:.2f}" for j in range(len(feature_names))])
        print(f"Sample {i}: {features_str}")
        print(f"  Actual {target_col} = ${y[i]:.2f}, Predicted {target_col} = ${y_pred[i]:.2f}")
    
    # Make predictions for new data points
    print("\n===== PREDICTIONS FOR NEW DATA =====")
    
    # Create new data points with varying values for each feature
    new_data = [
        [1500, 8000, 2000, 7],   # Small house, medium lot, newer, good quality
        [2500, 10000, 1980, 8],  # Medium house, large lot, older, very good quality
        [3500, 12000, 2010, 9],  # Large house, large lot, newer, excellent quality
        [1800, 7000, 1950, 5],   # Small-medium house, small lot, old, average quality
        [3000, 15000, 2020, 10]  # Large house, very large lot, newest, best quality
    ]
    
    # Normalize new data
    new_X_norm = []
    for sample in new_data:
        normalized_sample = []
        for j, value in enumerate(sample):
            normalized_sample.append((value - X_means[j]) / X_stds[j])
        new_X_norm.append(normalized_sample)
    
    # Make predictions (in normalized space)
    new_pred_norm = best_model.predict(new_X_norm)
    
    # Denormalize predictions
    new_pred = denormalize_predictions(new_pred_norm, y_mean, y_std)
    
    # Print predictions with feature values
    for i, sample in enumerate(new_data):
        features_str = ", ".join([f"{feature_names[j]} = {sample[j]}" for j in range(len(feature_names))])
        print(f"House {i+1}: {features_str}")
        print(f"  Predicted {target_col}: ${new_pred[i]:.2f}")
    
    # Plot actual vs predicted values for the most dominant feature
    most_dominant_feature = feature_names[0]  # Will be updated after analysis
    most_dominant_idx = 0
    
    # Find the most dominant feature from the analysis
    feature_importance = {}
    for i, feature in enumerate(feature_names):
        linear_coef = abs(best_model.coefficients[i + 1])
        higher_order_coefs = []
        for power in range(2, best_degree + 1):
            idx = len(feature_names) * (power - 1) + i + 1
            if idx < len(best_model.coefficients):
                higher_order_coefs.append(abs(best_model.coefficients[idx]))
        total_importance = linear_coef + sum(higher_order_coefs)
        feature_importance[feature] = total_importance
        
    # Find the most dominant feature
    most_dominant_feature = max(feature_importance.items(), key=lambda x: x[1])[0]
    most_dominant_idx = feature_names.index(most_dominant_feature)
    
    # Extract the values of the most dominant feature
    X_dominant = [sample[most_dominant_idx] for sample in X]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(X_dominant, y, alpha=0.5, label='Actual')
    
    # Sort by the dominant feature for smooth curve plotting
    # Create a list of (dominant_feature_value, full_feature_vector) pairs
    X_with_dominant = [(X[i][most_dominant_idx], X[i]) for i in range(len(X))]
    X_sorted_pairs = sorted(X_with_dominant, key=lambda x: x[0])
    
    # Extract sorted dominant feature values and corresponding full feature vectors
    X_dominant_sorted = [pair[0] for pair in X_sorted_pairs]
    X_full_sorted = [pair[1] for pair in X_sorted_pairs]
    
    # Normalize the sorted full feature vectors
    X_full_sorted_norm = []
    for sample in X_full_sorted:
        normalized_sample = []
        for j, value in enumerate(sample):
            normalized_sample.append((value - X_means[j]) / X_stds[j])
        X_full_sorted_norm.append(normalized_sample)
    
    # Predict using the sorted normalized full feature vectors
    y_sorted_norm = best_model.predict(X_full_sorted_norm)
    y_sorted = denormalize_predictions(y_sorted_norm, y_mean, y_std)
    
    plt.plot(X_dominant_sorted, y_sorted, 'r-', linewidth=2, label=f'Polynomial (degree {best_degree})')
    
    plt.title(f'Housing Prices vs {most_dominant_feature}')
    plt.xlabel(f'{most_dominant_feature}')
    plt.ylabel(f'{target_col} (in dollars)')
    plt.legend()
    plt.grid(True)
    plt.savefig('housing_polynomial_regression.png')
    
    # Save model parameters for later use in prediction
    model_params = {
        'degree': best_degree,
        'coefficients': best_model.coefficients,
        'X_means': X_means,
        'X_stds': X_stds,
        'y_mean': y_mean,
        'y_std': y_std,
        'feature_names': feature_names,
        'target_col': target_col
    }
    
    # Save model parameters to a CSV file
    with open('polynomial_model_params.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['degree', best_degree])
        writer.writerow(['y_mean', y_mean])
        writer.writerow(['y_std', y_std])
        writer.writerow(['target_col', target_col])
        
        # Write feature names
        for i, feature in enumerate(feature_names):
            writer.writerow([f'feature_name_{i}', feature])
        
        # Write feature means and stds
        for i, (mean, std) in enumerate(zip(X_means, X_stds)):
            writer.writerow([f'X_mean_{i}', mean])
            writer.writerow([f'X_std_{i}', std])
        
        # Write coefficients
        for i, coef in enumerate(best_model.coefficients):
            writer.writerow([f'coefficient_{i}', coef])

if __name__ == "__main__":
    main()
