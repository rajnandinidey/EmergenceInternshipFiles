import csv
import os
import numpy as np

def load_predictions(file_path):
    """Load predictions from a CSV file"""
    ids = []
    predictions = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header row
        
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    ids.append(int(row[0]))
                    predictions.append(float(row[1]))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse row: {row}")
    
    return ids, predictions

def main():
    print("Creating stacked model predictions...")
    
    # Define the models to include in the stack
    lasso_models = [
        "lasso_predictions_alpha_0.01.csv",
        "lasso_predictions_alpha_0.1.csv",
        "lasso_predictions_alpha_0.5.csv"
    ]
    
    ridge_models = [
        "ridge_predictions_alpha_0.01.csv",
        "ridge_predictions_alpha_0.1.csv",
        "ridge_predictions_alpha_0.5.csv"
    ]
    
    polynomial_models = [
        "housing_price_predictions_polynomial.csv"
    ]
    
    # Dictionary to store all predictions by ID
    all_predictions = {}
    
    # Load Lasso predictions
    print("Loading Lasso predictions...")
    for model_file in lasso_models:
        if os.path.exists(model_file):
            ids, predictions = load_predictions(model_file)
            
            for i, id_val in enumerate(ids):
                if id_val not in all_predictions:
                    all_predictions[id_val] = []
                all_predictions[id_val].append(predictions[i])
            
            print(f"  Loaded {len(predictions)} predictions from {model_file}")
        else:
            print(f"  Warning: {model_file} not found")
    
    # Load Ridge predictions
    print("Loading Ridge predictions...")
    for model_file in ridge_models:
        if os.path.exists(model_file):
            ids, predictions = load_predictions(model_file)
            
            for i, id_val in enumerate(ids):
                if id_val not in all_predictions:
                    all_predictions[id_val] = []
                all_predictions[id_val].append(predictions[i])
            
            print(f"  Loaded {len(predictions)} predictions from {model_file}")
        else:
            print(f"  Warning: {model_file} not found")
    
    # Load Polynomial predictions
    print("Loading Polynomial predictions...")
    for model_file in polynomial_models:
        if os.path.exists(model_file):
            # Polynomial predictions have a different format with GrLivArea column
            ids = []
            predictions = []
            
            with open(model_file, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # Skip header row
                
                for row in csv_reader:
                    if len(row) >= 3:  # Id, GrLivArea, SalePrice
                        try:
                            ids.append(int(row[0]))
                            predictions.append(float(row[2]))  # SalePrice is in column 3
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse row: {row}")
            
            for i, id_val in enumerate(ids):
                if id_val not in all_predictions:
                    all_predictions[id_val] = []
                all_predictions[id_val].append(predictions[i])
            
            print(f"  Loaded {len(predictions)} predictions from {model_file}")
        else:
            print(f"  Warning: {model_file} not found")
    
    # Calculate average predictions
    final_predictions = {}
    for id_val, preds in all_predictions.items():
        if preds:
            # Simple average of all predictions
            final_predictions[id_val] = sum(preds) / len(preds)
    
    # Sort by ID for consistent output
    sorted_ids = sorted(final_predictions.keys())
    
    # Write final predictions to CSV
    output_file = "housing_final_stacked_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for id_val in sorted_ids:
            writer.writerow([id_val, round(final_predictions[id_val], 4)])
    
    print(f"Stacked predictions saved to {output_file} with {len(final_predictions)} predictions")

if __name__ == "__main__":
    main()
