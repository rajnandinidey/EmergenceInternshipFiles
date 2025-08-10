import csv
import datetime
from linear_regression import linear_regression_gradient_descent, calculate_mse, predict
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path, max_rows=None):
    """
    Load data from CSV file
    
    Parameters:
    file_path : path to the CSV file
    max_rows : maximum number of rows to load (None for all)
    
    Returns:
    dates : list of datetime objects
    store_nbrs : list of store numbers
    families : list of product families
    sales : list of sales values
    onpromotion : list of promotion indicators
    """
    dates = []
    store_nbrs = []
    families = []
    sales = []
    onpromotion = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        
        row_count = 0
        for row in csv_reader:
            if len(row) >= 6:  # Ensure row has enough columns
                try:
                    dates.append(datetime.datetime.strptime(row[1], '%Y-%m-%d'))
                    store_nbrs.append(int(row[2]))
                    families.append(row[3])
                    sales.append(float(row[4]))
                    onpromotion.append(int(row[5]))
                    
                    row_count += 1
                    if max_rows and row_count >= max_rows:
                        break
                except (ValueError, IndexError) as e:
                    print(f"Skipping row due to error: {e}")
                    continue
    
    print(f"Loaded {len(sales)} rows of data")
    return dates, store_nbrs, families, sales, onpromotion

def normalize_data(X, y):
    """
    Normalize features and target values to prevent numerical overflow
    
    Parameters:
    X : list of feature values
    y : list of target values
    
    Returns:
    X_norm : normalized features
    y_norm : normalized target values
    X_mean, X_std : mean and standard deviation of X
    y_mean, y_std : mean and standard deviation of y
    """
    X_mean = sum(X) / len(X)
    X_std = (sum((x - X_mean) ** 2 for x in X) / len(X)) ** 0.5
    X_std = max(X_std, 1e-8)  # Prevent division by zero
    
    y_mean = sum(y) / len(y)
    y_std = (sum((val - y_mean) ** 2 for val in y) / len(y)) ** 0.5
    y_std = max(y_std, 1e-8)  # Prevent division by zero
    
    X_norm = [(x - X_mean) / X_std for x in X]
    y_norm = [(val - y_mean) / y_std for val in y]
    
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std

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

def filter_outliers(X, y, threshold=3):
    """
    Filter outliers based on z-score
    
    Parameters:
    X : list of feature values
    y : list of target values
    threshold : z-score threshold for outliers
    
    Returns:
    X_filtered, y_filtered : filtered data without outliers
    """
    # Calculate z-scores for y values
    y_mean = sum(y) / len(y)
    y_std = (sum((val - y_mean) ** 2 for val in y) / len(y)) ** 0.5
    
    if y_std == 0:
        return X, y  # No variation in y, return original data
    
    z_scores = [(val - y_mean) / y_std for val in y]
    
    # Filter out points with z-score above threshold
    filtered_indices = [i for i, z in enumerate(z_scores) if abs(z) <= threshold]
    
    X_filtered = [X[i] for i in filtered_indices]
    y_filtered = [y[i] for i in filtered_indices]
    
    print(f"Filtered out {len(y) - len(y_filtered)} outliers from {len(y)} points")
    return X_filtered, y_filtered

def prepare_features(dates):
    """
    Prepare features for linear regression
    
    Parameters:
    dates : list of datetime objects
    
    Returns:
    X : list of feature values (days since start)
    """
    # Convert dates to days since the first date
    start_date = min(dates)
    days_since_start = [(date - start_date).days for date in dates]
    
    return days_since_start

def main():
    # Load data (limit to 50,000 rows to prevent memory issues)
    print("Loading data...")
    file_path = "trainstoresales.csv"
    dates, store_nbrs, families, sales, onpromotion = load_data(file_path, max_rows=50000)
    
    # Prepare features
    print("Preparing features...")
    X = prepare_features(dates)
    y = sales
    
    # Filter outliers
    print("Filtering outliers...")
    X, y = filter_outliers(X, y)
    
    # Normalize data
    print("Normalizing data...")
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_data(X, y)
    
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
    print(f"Linear Regression Model: y = {slope:.6f}x + {intercept:.6f}")
    
    # Calculate and print MSE in original scale
    y_pred = [slope * x + intercept for x in X]
    mse = sum([(y[i] - y_pred[i])**2 for i in range(len(y))]) / len(y)
    print(f"Mean Squared Error: {mse:.6f}")
    
    # Print initial and final MSE from training (in normalized space)
    print(f"Initial MSE (normalized): {mse_history[0]:.6f}")
    print(f"Final MSE (normalized): {mse_history[-1]:.6f}")
    
    # Print some sample predictions
    print("\n===== SAMPLE PREDICTIONS =====")
    sample_indices = np.random.choice(len(X), 5, replace=False)
    for i in sample_indices:
        print(f"Day {X[i]}: Actual sales = {y[i]:.2f}, Predicted sales = {y_pred[i]:.2f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_history)), mse_history)
    plt.title('MSE during Training (Normalized Space)')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('training_progress.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    
    # Sample 1000 points for visualization if data is too large
    if len(X) > 1000:
        indices = np.random.choice(len(X), 1000, replace=False)
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        y_pred_sample = [y_pred[i] for i in indices]
    else:
        X_sample = X
        y_sample = y
        y_pred_sample = y_pred
    
    plt.scatter(X_sample, y_sample, alpha=0.5, label='Actual')
    plt.scatter(X_sample, y_pred_sample, alpha=0.5, label='Predicted')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Days since start')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png')
    
    print("\n===== TRAINING COMPLETED =====")
    print("Results saved to training_progress.png and actual_vs_predicted.png")

if __name__ == "__main__":
    main()
