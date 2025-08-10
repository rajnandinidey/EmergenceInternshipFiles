import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
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
    else:
        print(f"  Result: Data does not appear to be normally distributed (reject H0)")
    
    return p_value > alpha

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
    plt.savefig(f'improved_plots/{feature}_distribution.png')
    plt.close()

def apply_log1p_transform(df, numeric_features):
    """Apply log1p transformation to selected features"""
    transformed_df = df.copy()
    transformation_info = {}
    
    for feature in numeric_features:
        if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
            # Original skewness
            original_data = df[feature].dropna()
            original_skewness = calculate_skewness(original_data)
            
            # Apply log1p transformation
            transformed_df[feature] = np.log1p(df[feature])
            
            # New skewness
            transformed_data = transformed_df[feature].dropna()
            new_skewness = calculate_skewness(transformed_data)
            
            # Store transformation info
            transformation_info[feature] = {
                'original_skewness': original_skewness,
                'transformed_skewness': new_skewness,
                'improvement': original_skewness - new_skewness
            }
            
            print(f"Transformed {feature}:")
            print(f"  Original skewness: {original_skewness:.4f}")
            print(f"  After log1p transform: {new_skewness:.4f}")
            print(f"  Improvement: {original_skewness - new_skewness:.4f}")
            
            # Create distribution plots
            plot_distribution(
                original_data, 
                feature, 
                transformed=True, 
                log_data=transformed_data
            )
    
    return transformed_df, transformation_info

def main():
    print("Applying improved log1p transformation to housing data...")
    
    # Create output directories if they don't exist
    if not os.path.exists('improved_plots'):
        os.makedirs('improved_plots')
    if not os.path.exists('improved_data'):
        os.makedirs('improved_data')
    
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
    
    # Get numeric columns
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Calculate skewness for each numeric feature
    skewness_data = {}
    for col in numeric_cols:
        if col != 'Id':  # Skip Id column
            data = train_df[col].dropna()
            if len(data) > 0:
                skewness = calculate_skewness(data)
                skewness_data[col] = skewness
    
    # Sort features by skewness
    sorted_features = sorted(skewness_data.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nFeatures sorted by absolute skewness:")
    print("-" * 50)
    print(f"{'Feature':<20} {'Skewness':>10}")
    print("-" * 50)
    
    for feature, skew in sorted_features:
        print(f"{feature:<20} {skew:>10.4f}")
    
    # Select features with significant skewness (absolute value > 0.5)
    features_to_transform = [feature for feature, skew in sorted_features if abs(skew) > 0.5]
    
    print(f"\nSelected {len(features_to_transform)} features for transformation:")
    for feature in features_to_transform:
        print(f"  - {feature} (skewness: {skewness_data[feature]:.4f})")
    
    # Apply log1p transformation to training data
    print("\nApplying log1p transformations to training data...")
    transformed_train_df, train_transform_info = apply_log1p_transform(train_df, features_to_transform)
    
    # Save transformed training data
    transformed_train_file = "improved_data/train_housingprices_log1p_transformed.csv"
    transformed_train_df.to_csv(transformed_train_file, index=False)
    print(f"\nTransformed training data saved to {transformed_train_file}")
    
    # Apply same transformations to test data if available
    if test_df is not None:
        print("\nApplying log1p transformations to test data...")
        transformed_test_df, _ = apply_log1p_transform(test_df, features_to_transform)
        
        # Save transformed test data
        transformed_test_file = "improved_data/test_housingprices_log1p_transformed.csv"
        transformed_test_df.to_csv(transformed_test_file, index=False)
        print(f"Transformed test data saved to {transformed_test_file}")
    
    # Create a summary of transformations
    print("\nSummary of transformations:")
    print("-" * 70)
    print(f"{'Feature':<20} {'Original Skewness':>20} {'Transformed Skewness':>20} {'Improvement':>15}")
    print("-" * 70)
    
    for feature, info in sorted(train_transform_info.items(), key=lambda x: x[1]['improvement'], reverse=True):
        print(f"{feature:<20} {info['original_skewness']:>20.4f} {info['transformed_skewness']:>20.4f} {info['improvement']:>15.4f}")
    
    print("\nImproved log1p transformation complete. Check the 'improved_plots' directory for distribution visualizations.")

if __name__ == "__main__":
    main()
