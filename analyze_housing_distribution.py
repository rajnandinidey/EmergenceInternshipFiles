import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

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

def calculate_skewness(data):
    """Calculate skewness of the data"""
    return stats.skew(data)

def calculate_kurtosis(data):
    """Calculate kurtosis of the data"""
    return stats.kurtosis(data)

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

def main():
    print("Analyzing housing data distribution...")
    
    # Load training data
    train_file_path = "train_housingprices.csv"
    train_headers, train_data = load_housing_data(train_file_path)
    
    # Find index of target (SalePrice) and other important features
    target_index = train_headers.index('SalePrice')
    
    # Key features to analyze
    features_to_analyze = [
        'LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea', 
        'TotalBsmtSF', '1stFlrSF', 'FullBath', 'BedroomAbvGr',
        'GarageArea', 'SalePrice'
    ]
    
    feature_indices = {}
    for feature in features_to_analyze:
        if feature in train_headers:
            feature_indices[feature] = train_headers.index(feature)
    
    # Extract valid data for each feature
    feature_data = {}
    for feature, index in feature_indices.items():
        feature_data[feature] = [
            row[index] for row in train_data 
            if index < len(row) and row[index] is not None and isinstance(row[index], (int, float))
        ]
    
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('distribution_plots'):
        os.makedirs('distribution_plots')
    
    # Analyze each feature
    print("\nDistribution Analysis:")
    print("-" * 50)
    
    for feature, data in feature_data.items():
        # Calculate statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        skewness = calculate_skewness(data)
        kurtosis = calculate_kurtosis(data)
        
        print(f"\nFeature: {feature}")
        print(f"  Count: {len(data)}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Standard Deviation: {std_val:.4f}")
        print(f"  Skewness: {skewness:.4f}")
        print(f"  Kurtosis: {kurtosis:.4f}")
        
        # Interpret skewness
        if abs(skewness) < 0.5:
            print("  Skewness Interpretation: Approximately symmetric")
        elif abs(skewness) < 1.0:
            direction = "right" if skewness > 0 else "left"
            print(f"  Skewness Interpretation: Moderately skewed to the {direction}")
        else:
            direction = "right" if skewness > 0 else "left"
            print(f"  Skewness Interpretation: Highly skewed to the {direction}")
        
        # Perform normality test
        is_normal = perform_normality_test(data, feature)
        
        # Create histogram with normal curve
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(data, kde=True, stat="density")
        
        # Plot normal distribution curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_val, std_val)
        plt.plot(x, p, 'k', linewidth=2)
        
        # Add vertical lines for mean and median
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
        
        # Add plot details
        plt.title(f'Distribution of {feature}\nSkewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(f'distribution_plots/{feature}_distribution.png')
        plt.close()
    
    # Create Q-Q plots for SalePrice
    plt.figure(figsize=(10, 6))
    stats.probplot(feature_data['SalePrice'], plot=plt)
    plt.title('Q-Q Plot for SalePrice')
    plt.grid(True, alpha=0.3)
    plt.savefig('distribution_plots/SalePrice_qq_plot.png')
    plt.close()
    
    # Try log transformation on SalePrice if it's skewed
    if abs(calculate_skewness(feature_data['SalePrice'])) > 0.5:
        log_sale_price = np.log1p(feature_data['SalePrice'])  # log(1+x) to handle zeros
        
        # Calculate statistics for log-transformed data
        log_skewness = calculate_skewness(log_sale_price)
        log_kurtosis = calculate_kurtosis(log_sale_price)
        
        print("\nLog-transformed SalePrice:")
        print(f"  Skewness: {log_skewness:.4f}")
        print(f"  Kurtosis: {log_kurtosis:.4f}")
        
        # Interpret skewness
        if abs(log_skewness) < 0.5:
            print("  Skewness Interpretation: Approximately symmetric")
        elif abs(log_skewness) < 1.0:
            direction = "right" if log_skewness > 0 else "left"
            print(f"  Skewness Interpretation: Moderately skewed to the {direction}")
        else:
            direction = "right" if log_skewness > 0 else "left"
            print(f"  Skewness Interpretation: Highly skewed to the {direction}")
        
        # Perform normality test on log-transformed data
        perform_normality_test(log_sale_price, "Log-transformed SalePrice")
        
        # Create histogram for log-transformed data
        plt.figure(figsize=(10, 6))
        sns.histplot(log_sale_price, kde=True, stat="density")
        plt.title(f'Distribution of Log-transformed SalePrice\nSkewness: {log_skewness:.4f}, Kurtosis: {log_kurtosis:.4f}')
        plt.xlabel('Log(SalePrice)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig('distribution_plots/SalePrice_log_distribution.png')
        plt.close()
        
        # Create Q-Q plot for log-transformed SalePrice
        plt.figure(figsize=(10, 6))
        stats.probplot(log_sale_price, plot=plt)
        plt.title('Q-Q Plot for Log-transformed SalePrice')
        plt.grid(True, alpha=0.3)
        plt.savefig('distribution_plots/SalePrice_log_qq_plot.png')
        plt.close()
    
    print("\nAnalysis complete. Distribution plots saved to 'distribution_plots' directory.")

if __name__ == "__main__":
    main()
