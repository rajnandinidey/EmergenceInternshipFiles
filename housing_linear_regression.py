import csv
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import linear_regression_gradient_descent, calculate_mse, predict

def load_housing_data(file_path, feature_col='GrLivArea', target_col='SalePrice'):
    """
    Load housing data from CSV file
    
    Parameters:
    file_path : path to the CSV file
    feature_col : name of the feature column
    target_col : name of the target column
    
    Returns:
    headers : list of column names
    data : list of lists containing the data with valid feature and target values
    """
    headers = []
    data = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get header row
        
        # Find the indices of the feature and target columns
        try:
            feature_idx = headers.index(feature_col)
            target_idx = headers.index(target_col)
        except ValueError:
            print(f"Error: Could not find {feature_col} or {target_col} in the CSV headers.")
            return headers, []
        
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
            
            # Only add rows where both feature and target values are valid numbers
            if (feature_idx < len(processed_row) and target_idx < len(processed_row) and
                processed_row[feature_idx] is not None and processed_row[target_idx] is not None):
                data.append(processed_row)
    
    print(f"Loaded {len(data)} rows with valid {feature_col} and {target_col} values")
    return headers, data

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

def denormalize_parameters(slope, intercept, X_mean, X_std, y_mean, y_std):
    """
    Convert normalized model parameters back to original scale
    
    Parameters:
    slope, intercept : model parameters in normalized space
    X_mean, X_std : mean and standard deviation of X
    y_mean, y_std : mean and standard deviation of y
    
    Returns:
    original_slope, original_intercept : model parameters in original scale
    """
    original_slope = slope * (y_std / X_std)
    original_intercept = y_mean - original_slope * X_mean
    
    return original_slope, original_intercept

def main():
    # Load data
    print("Loading housing data...")
    file_path = "train_housingprices.csv"
    feature_col = 'GrLivArea'
    target_col = 'SalePrice'
    headers, data = load_housing_data(file_path, feature_col, target_col)
    
    # Convert data to columns
    columns = list(zip(*data))
    
    # Get feature names and their indices
    feature_names = headers[:-1]  # All columns except the last one (MEDV)
    
    print("\nAvailable features:")
    for i, feature in enumerate(feature_names):
        print(f"{i+1}. {feature}")
    
    # For this example, we'll use GrLivArea (Above grade (ground) living area square feet) to predict SalePrice
    # Find the indices of the features we want to use
    feature_index = headers.index('GrLivArea') if 'GrLivArea' in headers else 46  # GrLivArea is typically at index 46
    feature_name = headers[feature_index]
    target_index = len(headers) - 1  # SalePrice is the last column
    
    print(f"\nUsing {feature_name} to predict {headers[target_index]}")
    
    # Extract feature and target values
    X = [row[feature_index] for row in data]
    y = [row[target_index] for row in data]
    
    # Normalize data
    print("Normalizing data...")
    X_norm, X_mean, X_std = normalize_data(X)
    y_norm, y_mean, y_std = normalize_data(y)
    
    # Train the model with normalized data
    print("Training linear regression model...")
    slope_norm, intercept_norm, mse_history = linear_regression_gradient_descent(
        X_norm, y_norm, learning_rate=0.01, iterations=1000
    )
    
    # Convert parameters back to original scale
    slope, intercept = denormalize_parameters(
        slope_norm, intercept_norm, X_mean, X_std, y_mean, y_std
    )
    
    # Print model parameters
    print("\n===== MODEL RESULTS =====")
    print(f"Linear Regression Model: {headers[target_index]} = {slope:.6f} × {feature_name} + {intercept:.6f}")
    
    # Calculate and print MSE in original scale
    y_pred = [slope * x + intercept for x in X]
    mse = sum([(y[i] - y_pred[i])**2 for i in range(len(y))]) / len(y)
    print(f"Mean Squared Error: {mse:.6f}")
    
    # Calculate R-squared
    y_mean_val = sum(y) / len(y)
    ss_total = sum([(val - y_mean_val) ** 2 for val in y])
    ss_residual = sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))])
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R-squared: {r_squared:.6f}")
    
    # Print initial and final MSE from training (in normalized space)
    print(f"Initial MSE (normalized): {mse_history[0]:.6f}")
    print(f"Final MSE (normalized): {mse_history[-1]:.6f}")
    
    # Print some sample predictions
    print("\n===== SAMPLE PREDICTIONS =====")
    sample_indices = np.random.choice(len(X), 5, replace=False)
    for i in sample_indices:
        print(f"{feature_name} = {X[i]:.2f}: Actual {headers[target_index]} = {y[i]:.2f}, Predicted {headers[target_index]} = {y_pred[i]:.2f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_history)), mse_history)
    plt.title('MSE during Training (Normalized Space)')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('housing_training_progress.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, alpha=0.5, label='Actual')
    
    # Generate points for the regression line
    x_line = np.linspace(min(X), max(X), 100)
    y_line = [slope * x + intercept for x in x_line]
    plt.plot(x_line, y_line, 'r-', label='Regression Line')
    
    plt.title(f'Housing Prices vs {feature_name}')
    plt.xlabel(f'{feature_name} (Above grade living area in square feet)')
    plt.ylabel(f'{headers[target_index]} (in dollars)')
    plt.legend()
    plt.grid(True)
    plt.savefig('housing_regression.png')
    
    print("\n===== TRAINING COMPLETED =====")
    print("Results saved to housing_training_progress.png and housing_regression.png")

if __name__ == "__main__":
    main()
