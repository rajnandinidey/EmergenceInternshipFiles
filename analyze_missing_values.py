import pandas as pd
import numpy as np

def analyze_missing_values(file_path):
    """
    Analyze a dataset to identify features with missing values
    and count the number of missing values in each feature.
    """
    print(f"Analyzing missing values in {file_path}...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check for missing values in each column
    missing_counts = {}
    for col in df.columns:
        # Count missing values
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_counts[col] = missing_count
    
    # Display results
    if missing_counts:
        print("\nFeatures with missing values:")
        print("-" * 50)
        print(f"{'Feature':<30} {'Count of Missing Values':<20} {'Percentage':<10}")
        print("-" * 50)
        
        for col, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / df.shape[0]) * 100
            print(f"{col:<30} {count:<20} {percentage:.2f}%")
        
        # Total missing values
        total_missing = sum(missing_counts.values())
        total_cells = df.shape[0] * df.shape[1]
        print("\nSummary:")
        print(f"Total missing values: {total_missing}")
        print(f"Total cells in dataset: {total_cells}")
        print(f"Percentage of missing values: {(total_missing / total_cells) * 100:.2f}%")
    else:
        print("\nNo missing values found in the dataset.")

def main():
    # Analyze both training and test datasets
    train_file = "train_housingprices.csv"
    test_file = "test_housingprices.csv"
    
    print("=== TRAINING DATASET ANALYSIS ===")
    analyze_missing_values(train_file)
    
    print("\n\n=== TEST DATASET ANALYSIS ===")
    analyze_missing_values(test_file)

if __name__ == "__main__":
    main()
