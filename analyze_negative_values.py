import pandas as pd
import numpy as np

def analyze_negative_values(file_path):
    """
    Analyze a dataset to identify features with negative values
    and count the number of negative values in each feature.
    """
    print(f"Analyzing negative values in {file_path}...")
    
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
    
    # Check for negative values in each numeric column
    negative_counts = {}
    for col in numeric_cols:
        # Count negative values
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_counts[col] = neg_count
    
    # Display results
    if negative_counts:
        print("\nFeatures with negative values:")
        print("-" * 50)
        print(f"{'Feature':<30} {'Count of Negative Values':<20} {'Percentage':<10}")
        print("-" * 50)
        
        for col, count in sorted(negative_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / df.shape[0]) * 100
            print(f"{col:<30} {count:<20} {percentage:.2f}%")
        
        # Show some examples of rows with negative values
        print("\nExample rows with negative values:")
        for col in list(negative_counts.keys())[:3]:  # Show examples for up to 3 features
            print(f"\nRows with negative values in '{col}':")
            neg_rows = df[df[col] < 0].head(5)  # Show up to 5 rows
            print(neg_rows[['Id', col]].to_string(index=False))
    else:
        print("\nNo features with negative values found in the dataset.")

def main():
    # Analyze both training and test datasets
    train_file = "train_housingprices.csv"
    test_file = "test_housingprices.csv"
    
    print("=== TRAINING DATASET ANALYSIS ===")
    analyze_negative_values(train_file)
    
    print("\n\n=== TEST DATASET ANALYSIS ===")
    analyze_negative_values(test_file)

if __name__ == "__main__":
    main()
