import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import os

def load_housing_data(file_path):
    """Load housing data from CSV file into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_skewness(data):
    """Calculate skewness of the data"""
    return stats.skew(data)

def perform_normality_test(data, feature_name):
    """Perform Shapiro-Wilk test for normality"""
    # Sample data if there are too many points (Shapiro-Wilk has a limit)
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)
    
    stat, p_value = stats.shapiro(data)
    alpha = 0.05
    
    print(f"Shapiro-Wilk Test for {feature_name}:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    
    if p_value > alpha:
        print(f"  Result: Data appears to be normally distributed (fail to reject H0)")
        return True
    else:
        print(f"  Result: Data does not appear to be normally distributed (reject H0)")
        return False

def plot_distribution(data, feature, transformed=False, log_data=None):
    """Create and save distribution plots"""
    plt.figure(figsize=(12, 5))
    
    # Original data plot
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, stat="density")
    
    # Add normal curve
    mean_val = np.mean(data)
    std_val = np.std(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_val, std_val)
    plt.plot(x, p, 'k', linewidth=2)
    
    skewness = calculate_skewness(data)
    title = f'Original {feature}\nSkewness: {skewness:.4f}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Transformed data plot if available
    if transformed and log_data is not None:
        plt.subplot(1, 2, 2)
        sns.histplot(log_data, kde=True, stat="density")
        
        # Add normal curve
        mean_val = np.mean(log_data)
        std_val = np.std(log_data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_val, std_val)
        plt.plot(x, p, 'k', linewidth=2)
        
        log_skewness = calculate_skewness(log_data)
        title = f'Log-transformed {feature}\nSkewness: {log_skewness:.4f}'
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'transformed_plots/{feature}_distribution.png')
    plt.close()

def apply_log_transform(df, features_to_transform):
    """Apply log transformation to selected features"""
    transformed_df = df.copy()
    transformation_info = {}
    
    for feature in features_to_transform:
        if feature in df.columns:
            # Check if feature contains valid numeric data
            if df[feature].dtype in ['int64', 'float64'] and (df[feature] > 0).all():
                # Original skewness
                original_skewness = calculate_skewness(df[feature].dropna())
                
                # Apply log transformation (log1p to handle small values better)
                transformed_df[feature] = np.log1p(df[feature])
                
                # New skewness
                new_skewness = calculate_skewness(transformed_df[feature].dropna())
                
                # Store transformation info
                transformation_info[feature] = {
                    'original_skewness': original_skewness,
                    'transformed_skewness': new_skewness,
                    'improvement': original_skewness - new_skewness
                }
                
                print(f"Transformed {feature}:")
                print(f"  Original skewness: {original_skewness:.4f}")
                print(f"  After log transform: {new_skewness:.4f}")
                print(f"  Improvement: {original_skewness - new_skewness:.4f}")
                
                # Create distribution plots
                plot_distribution(
                    df[feature].dropna(), 
                    feature, 
                    transformed=True, 
                    log_data=transformed_df[feature].dropna()
                )
            elif feature == 'SalePrice' and 'SalePrice' in df.columns:
                # Special handling for SalePrice which might be in test data but with NaN values
                print(f"Note: {feature} found but contains invalid values for transformation")
            else:
                print(f"Warning: {feature} contains zero or negative values, cannot apply log transform")
        else:
            print(f"Warning: {feature} not found in dataset")
    
    return transformed_df, transformation_info

def main():
    print("Applying log transformation to make housing data more normally distributed...")
    
    # Create output directories if they don't exist
    if not os.path.exists('transformed_plots'):
        os.makedirs('transformed_plots')
    if not os.path.exists('transformed_data'):
        os.makedirs('transformed_data')
    
    # Load training data
    train_file_path = "train_housingprices.csv"
    train_df = load_housing_data(train_file_path)
    
    if train_df is None:
        print("Failed to load training data. Exiting.")
        return
    
    print(f"Loaded training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    
    # Load test data
    test_file_path = "test_housingprices.csv"
    test_df = load_housing_data(test_file_path)
    
    if test_df is None:
        print("Failed to load test data. Continuing with just training data.")
    else:
        print(f"Loaded test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Features to transform based on skewness analysis
    features_to_transform = [
        'LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 
        'SalePrice', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal', 'GarageArea'
    ]
    
    # Apply transformations to training data
    print("\nApplying log transformations to training data...")
    transformed_train_df, train_transform_info = apply_log_transform(train_df, features_to_transform)
    
    # Save transformed training data
    transformed_train_file = "transformed_data/train_housingprices_log_transformed.csv"
    transformed_train_df.to_csv(transformed_train_file, index=False)
    print(f"\nTransformed training data saved to {transformed_train_file}")
    
    # Apply same transformations to test data if available
    if test_df is not None:
        print("\nApplying log transformations to test data...")
        transformed_test_df, _ = apply_log_transform(test_df, [f for f in features_to_transform if f != 'SalePrice'])
        
        # Save transformed test data
        transformed_test_file = "transformed_data/test_housingprices_log_transformed.csv"
        transformed_test_df.to_csv(transformed_test_file, index=False)
        print(f"Transformed test data saved to {transformed_test_file}")
    
    # Create a summary of transformations
    print("\nSummary of transformations:")
    print("-" * 60)
    print(f"{'Feature':<15} {'Original Skewness':>20} {'Transformed Skewness':>20} {'Improvement':>15}")
    print("-" * 60)
    
    for feature, info in sorted(train_transform_info.items(), key=lambda x: x[1]['improvement'], reverse=True):
        print(f"{feature:<15} {info['original_skewness']:>20.4f} {info['transformed_skewness']:>20.4f} {info['improvement']:>15.4f}")
    
    print("\nLog transformation complete. Check the 'transformed_plots' directory for distribution visualizations.")

if __name__ == "__main__":
    main()
