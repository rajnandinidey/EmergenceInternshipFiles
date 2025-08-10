def linear_regression_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Simple linear regression using gradient descent to minimize MSE
    
    Parameters:
    X : list of numbers - independent variable
    y : list of numbers - dependent variable
    learning_rate : learning rate for gradient descent
    iterations : number of iterations for gradient descent
    
    Returns:
    slope, intercept - the parameters of the linear model
    mse_history - list of MSE values during training
    """
    n = len(X)
    
    # Initialize parameters
    slope = 0
    intercept = 0
    mse_history = []
    
    # Gradient descent
    for i in range(iterations):
        # Calculate predictions
        y_pred = [slope * x + intercept for x in X]
        
        # Calculate MSE
        mse = sum([(y[j] - y_pred[j])**2 for j in range(n)]) / n
        mse_history.append(mse)
        
        # Calculate gradients
        gradient_slope = (-2/n) * sum([(y[j] - y_pred[j]) * X[j] for j in range(n)])
        gradient_intercept = (-2/n) * sum([(y[j] - y_pred[j]) for j in range(n)])
        
        # Update parameters
        slope = slope - learning_rate * gradient_slope
        intercept = intercept - learning_rate * gradient_intercept
    
    # Calculate final MSE
    y_pred = [slope * x + intercept for x in X]
    final_mse = sum([(y[j] - y_pred[j])**2 for j in range(n)]) / n
    mse_history.append(final_mse)
    
    return slope, intercept, mse_history

def calculate_mse(X, y, slope, intercept):
    """
    Calculate Mean Squared Error
    
    Parameters:
    X : list of numbers - independent variable
    y : list of numbers - actual values
    slope : slope of the line
    intercept : y-intercept of the line
    
    Returns:
    mse : Mean Squared Error
    """
    n = len(X)
    y_pred = [slope * x + intercept for x in X]
    mse = sum([(y[i] - y_pred[i])**2 for i in range(n)]) / n
    return mse

def predict(X, slope, intercept):
   
    # Make predictions using the linear model
    
    # X : number or list of numbers
    # slope : slope of the line
    # intercept : y-intercept of the line
    
    # Returns: Predicted values
    
    if isinstance(X, (int, float)):
        return slope * X + intercept
    return [slope * x + intercept for x in X]


import csv
import matplotlib.pyplot as plt
import numpy as np

def load_housing_data(file_path):
    """
    Load housing data from CSV file
    
    Parameters:
    file_path : path to the CSV file
    
    Returns:
    headers : list of column names
    data : list of lists containing the data
    """
    headers = []
    data = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get header row
        
        for row in csv_reader:
            # Replace 'NA' values with None
            processed_row = []
            for value in row:
                if value == 'NA':
                    processed_row.append(None)
                else:
                    try:
                        processed_row.append(float(value))
                    except ValueError:
                        processed_row.append(None)
            
            # Only add rows that have all values
            if None not in processed_row:
                data.append(processed_row)
    
    print(f"Loaded {len(data)} complete rows of data")
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

# Housing data implementation
if __name__ == "__main__":
    # Load housing data
    print("Loading housing data...")
    file_path = "HousingData.csv"
    headers, data = load_housing_data(file_path)
    
    # Print available features
    feature_names = headers[:-1]  # All columns except the last one (MEDV)
    print("\nAvailable features:")
    for i, feature in enumerate(feature_names):
        print(f"{i+1}. {feature}")
    
    # For this example, we'll use RM (average number of rooms) to predict MEDV (median home value)
    # RM is at index 5 in the data
    feature_index = 5
    feature_name = headers[feature_index]
    target_index = len(headers) - 1  # MEDV is the last column
    
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
    print(f"Linear Regression Model: {headers[target_index]} = {slope:.4f} × {feature_name} + {intercept:.4f}")
    
    # Calculate and print MSE in original scale
    y_pred = [slope * x + intercept for x in X]
    mse = sum([(y[i] - y_pred[i])**2 for i in range(len(y))]) / len(y)
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Calculate R-squared
    y_mean_val = sum(y) / len(y)
    ss_total = sum([(val - y_mean_val) ** 2 for val in y])
    ss_residual = sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))])
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R-squared: {r_squared:.4f}")
    
    # Print initial and final MSE from training (in normalized space)
    print(f"Initial MSE (normalized): {mse_history[0]:.4f}")
    print(f"Final MSE (normalized): {mse_history[-1]:.4f}")
    
    # Print some sample predictions
    print("\n===== SAMPLE PREDICTIONS =====")
    sample_indices = np.random.choice(len(X), 5, replace=False)
    for i in sample_indices:
        print(f"{feature_name} = {X[i]:.2f}: Actual {headers[target_index]} = {y[i]:.2f}, Predicted {headers[target_index]} = {y_pred[i]:.2f}")
    
    # Make predictions for new data points
    print("\n===== PREDICTIONS FOR NEW DATA =====")
    new_X = [5.0, 6.0, 7.0, 8.0]  # New room values to predict
    new_predictions = [slope * x + intercept for x in new_X]
    for i, x in enumerate(new_X):
        print(f"For a house with {x:.1f} rooms, predicted {headers[target_index]}: ${new_predictions[i]:.2f}k")
    
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
    plt.xlabel(f'{feature_name} (Average Number of Rooms)')
    plt.ylabel(f'{headers[target_index]} (Median Home Value in $1000s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('housing_regression.png')
    
    print("\n===== TRAINING COMPLETED =====")
    print("Results saved to housing_training_progress.png and housing_regression.png")
