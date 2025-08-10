import numpy as np
import pandas as pd
import os
import glob

def load_predictions(file_path):
    """Load predictions from a CSV file"""
    predictions = {}
    try:
        with open(file_path, 'r') as file:
            import csv
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if len(row) >= 2:
                    id_val = int(row[0])
                    pred = float(row[1])
                    predictions[id_val] = pred
        return predictions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def main():
    print("Creating final stacked predictions including Random Forest models...")
    
    # Find all prediction files
    prediction_files = []
    
    # Find all optimized model prediction files
    lasso_files = glob.glob("optimized_lasso_*_predictions.csv")
    ridge_files = glob.glob("optimized_ridge_*_predictions.csv")
    poly_files = glob.glob("optimized_poly_*_predictions.csv")
    tree_files = glob.glob("optimized_tree_*_predictions.csv")
    rf_files = glob.glob("optimized_rf_*_predictions.csv")
    
    prediction_files = lasso_files + ridge_files + poly_files + tree_files + rf_files
    
    print(f"Found {len(prediction_files)} prediction files:")
    for file in prediction_files:
        print(f"  - {file}")
    
    # Load all predictions
    all_predictions = {}
    
    for file_path in prediction_files:
        file_predictions = load_predictions(file_path)
        
        for id_val, pred in file_predictions.items():
            if id_val not in all_predictions:
                all_predictions[id_val] = []
            all_predictions[id_val].append(pred)
    
    # Calculate average predictions
    final_predictions = {}
    for id_val, preds in all_predictions.items():
        if preds:
            final_predictions[id_val] = sum(preds) / len(preds)
    
    # Save final stacked predictions
    output_file = "final_stacked_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for id_val in sorted(final_predictions.keys()):
            pred = final_predictions[id_val]
            writer.writerow([id_val, round(pred, 4)])
    
    print(f"\nFinal stacked predictions saved to {output_file}")
    print(f"Included {len(prediction_files)} models in the ensemble")
    
    # Calculate model type weights in the ensemble
    model_counts = {
        'Lasso': len(lasso_files),
        'Ridge': len(ridge_files),
        'Polynomial': len(poly_files),
        'Decision Tree': len(tree_files),
        'Random Forest': len(rf_files)
    }
    
    print("\nModel composition in the ensemble:")
    for model_type, count in model_counts.items():
        percentage = (count / len(prediction_files)) * 100
        print(f"  - {model_type}: {count} models ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
