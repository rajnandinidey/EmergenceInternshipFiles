import pandas as pd
import numpy as np

def analyze_zero_values(file_path):
    """
    Analyze a dataset to identify features with zero values
    and count the number of zero values in each feature.
    """
    print(f"Analyzing zero values in {file_path}...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric columns")
    
    # Check for zero values in each numeric column
    zero_counts = {}
    for col in numeric_cols:
        # Count zero values
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            zero_counts[col] = zero_count
    
    # Display results
    if zero_counts:
        print("\nFeatures with zero values:")
        print("-" * 50)
        print(f"{'Feature':<30} {'Count of Zero Values':<20} {'Percentage':<10}")
        print("-" * 50)
        
        for col, count in sorted(zero_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / df.shape[0]) * 100
            print(f"{col:<30} {count:<20} {percentage:.2f}%")
    else:
        print("\nNo features with zero values found in the dataset.")

def main():
    # Analyze both training and test datasets
    train_file = "train_housingprices.csv"
    test_file = "test_housingprices.csv"
    
    print("=== TRAINING DATASET ANALYSIS ===")
    analyze_zero_values(train_file)
    
    print("\n\n=== TEST DATASET ANALYSIS ===")
    analyze_zero_values(test_file)

if __name__ == "__main__":
    main()
