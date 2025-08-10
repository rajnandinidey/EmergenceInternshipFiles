import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions(file_path):
    """Load predictions from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    print("Comparing Random Forest predictions from different approaches...")
    
    # Load predictions from different approaches
    files = {
        "RF Mean Imputed (100 trees, depth 15)": "mean_imputed_rf_100trees_depth15_predictions.csv",
        "RF Mean Imputed (200 trees, unlimited depth)": "mean_imputed_rf_200trees_depthNone_predictions.csv",
        "RF Optimized Transform (100 trees, depth 15)": "optimized_rf_100trees_depth15_predictions.csv",
        "RF Optimized Transform (200 trees, unlimited depth)": "optimized_rf_200trees_depthNone_predictions.csv",
        "Final Ensemble": "final_ensemble_predictions.csv"
    }
    
    # Load all prediction files
    predictions = {}
    for name, file_path in files.items():
        df = load_predictions(file_path)
        if df is not None:
            predictions[name] = df
            print(f"Loaded {name} with {len(df)} predictions")
        else:
            print(f"Failed to load {name}")
    
    # Calculate statistics for each prediction set
    stats = {}
    for name, df in predictions.items():
        if 'SalePrice' in df.columns:
            stats[name] = {
                'mean': df['SalePrice'].mean(),
                'median': df['SalePrice'].median(),
                'min': df['SalePrice'].min(),
                'max': df['SalePrice'].max(),
                'std': df['SalePrice'].std()
            }
    
    # Display statistics
    print("\nPrediction Statistics:")
    print("-" * 80)
    print(f"{'Model':<40} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std Dev':<12}")
    print("-" * 80)
    for name, stat in stats.items():
        print(f"{name:<40} {stat['mean']:<12.2f} {stat['median']:<12.2f} "
              f"{stat['min']:<12.2f} {stat['max']:<12.2f} {stat['std']:<12.2f}")
    
    # Calculate correlation between different prediction sets
    print("\nCorrelation between prediction sets:")
    merged_df = pd.DataFrame({'Id': predictions[list(predictions.keys())[0]]['Id']})
    for name, df in predictions.items():
        merged_df[name] = df['SalePrice']
    
    correlation = merged_df.drop('Id', axis=1).corr()
    print(correlation)
    
    # Calculate differences between mean-imputed and optimized transform approaches
    if "RF Mean Imputed (100 trees, depth 15)" in predictions and "RF Optimized Transform (100 trees, depth 15)" in predictions:
        mean_imputed = predictions["RF Mean Imputed (100 trees, depth 15)"]
        optimized = predictions["RF Optimized Transform (100 trees, depth 15)"]
        
        # Merge on Id to compare
        comparison = pd.merge(mean_imputed, optimized, on='Id', suffixes=('_mean_imputed', '_optimized'))
        comparison['diff'] = comparison['SalePrice_mean_imputed'] - comparison['SalePrice_optimized']
        comparison['abs_diff'] = abs(comparison['diff'])
        comparison['pct_diff'] = (comparison['diff'] / comparison['SalePrice_optimized']) * 100
        
        print("\nDifferences between Mean Imputed and Optimized Transform (100 trees, depth 15):")
        print(f"Average absolute difference: ${comparison['abs_diff'].mean():.2f}")
        print(f"Average percentage difference: {comparison['pct_diff'].mean():.2f}%")
        print(f"Maximum absolute difference: ${comparison['abs_diff'].max():.2f}")
        print(f"Maximum percentage difference: {comparison['pct_diff'].abs().max():.2f}%")
        
        # Find houses with largest differences
        print("\nHouses with largest differences:")
        largest_diff = comparison.nlargest(5, 'abs_diff')
        print(largest_diff[['Id', 'SalePrice_mean_imputed', 'SalePrice_optimized', 'diff', 'pct_diff']])
    
    # Create a new ensemble with mean-imputed RF models
    if "RF Mean Imputed (100 trees, depth 15)" in predictions and "RF Mean Imputed (200 trees, unlimited depth)" in predictions:
        mean_imputed_100 = predictions["RF Mean Imputed (100 trees, depth 15)"]
        mean_imputed_200 = predictions["RF Mean Imputed (200 trees, unlimited depth)"]
        
        # Create ensemble of just the mean-imputed models
        ensemble = pd.merge(mean_imputed_100, mean_imputed_200, on='Id', suffixes=('_100', '_200'))
        ensemble['SalePrice'] = (ensemble['SalePrice_100'] + ensemble['SalePrice_200']) / 2
        
        # Save the ensemble predictions
        output_file = "mean_imputed_rf_ensemble_predictions.csv"
        ensemble[['Id', 'SalePrice']].to_csv(output_file, index=False)
        print(f"\nSaved mean-imputed RF ensemble predictions to {output_file}")

if __name__ == "__main__":
    main()
