import csv
import numpy as np
from polynomial_regression import PolynomialRegression

def load_housing_data(file_path, feature_col='GrLivArea'):
    """
    Load housing data from CSV file for prediction
    
    Parameters:
    file_path : path to the CSV file
    feature_col : name of the feature column
    
    Returns:
    headers : list of column names
    data : list of lists containing the data with valid feature values
    ids : list of house IDs
    """
    headers = []
    data = []
    ids = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get header row
        
        # Find the indices of the feature column and ID column
        try:
            feature_idx = headers.index(feature_col)
            id_idx = headers.index('Id')
        except ValueError:
            print(f"Error: Could not find {feature_col} or Id in the CSV headers.")
            return headers, [], []
        
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
            
            # Only add rows where feature value is valid
            if (feature_idx < len(processed_row) and 
                processed_row[feature_idx] is not None):
                data.append(processed_row)
                ids.append(processed_row[id_idx])
    
    print(f"Loaded {len(data)} rows with valid {feature_col} values")
    return headers, data, ids

def load_model_params(file_path):
    """
    Load model parameters from CSV file
    
    Parameters:
    file_path : path to the CSV file
    
    Returns:
    model_params : dictionary of model parameters
    """
    model_params = {}
    coefficients = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        
        for row in csv_reader:
            param_name = row[0]
            param_value = row[1]
            
            if param_name.startswith('coefficient_'):
                coefficients.append(float(param_value))
            elif param_name in ['degree', 'X_mean', 'X_std', 'y_mean', 'y_std']:
                model_params[param_name] = float(param_value)
            else:
                model_params[param_name] = param_value
    
    model_params['coefficients'] = coefficients
    return model_params

def main():
    # Load model parameters
    print("Loading model parameters...")
    model_params = load_model_params('polynomial_model_params.csv')
    
    # Extract model parameters
    degree = int(model_params['degree'])
    coefficients = model_params['coefficients']
    X_mean = model_params['X_mean']
    X_std = model_params['X_std']
    y_mean = model_params['y_mean']
    y_std = model_params['y_std']
    feature_col = model_params.get('feature_col', 'GrLivArea')
    
    print(f"Loaded polynomial model with degree {degree}")
    print(f"Feature: {feature_col}")
    
    # Create polynomial regression model
    model = PolynomialRegression(degree=degree)
    model.coefficients = coefficients
    
    # Load test data
    print("\nLoading test housing data...")
    test_file_path = "test_housingprices.csv"
    headers, test_data, ids = load_housing_data(test_file_path, feature_col)
    
    # Find the index of the feature column
    feature_idx = headers.index(feature_col)
    
    # Extract feature values
    X_test = [row[feature_idx] for row in test_data]
    
    # Normalize test data
    X_test_norm = [(x - X_mean) / X_std for x in X_test]
    
    # Make predictions (in normalized space)
    print("Making predictions...")
    y_pred_norm = model.predict(X_test_norm)
    
    # Denormalize predictions
    predictions = [y * y_std + y_mean for y in y_pred_norm]
    
    # Create output data
    output_data = []
    for i in range(len(ids)):
        output_data.append({
            'Id': int(ids[i]),
            'GrLivArea': X_test[i],
            'SalePrice': predictions[i]
        })
    
    # Sort by Id
    output_data.sort(key=lambda x: x['Id'])
    
    # Save predictions to CSV
    output_file = 'housing_price_predictions_polynomial.csv'
    print(f"Saving predictions to {output_file}...")
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'GrLivArea', 'SalePrice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)
    
    print(f"Predictions saved to {output_file}")
    
    # Also create a submission file with just Id and SalePrice
    submission_file = 'submission_polynomial.csv'
    print(f"Creating submission file {submission_file}...")
    
    with open(submission_file, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'SalePrice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in output_data:
            writer.writerow({
                'Id': row['Id'],
                'SalePrice': row['SalePrice']
            })
    
    print(f"Submission file saved to {submission_file}")
    
    # Print some sample predictions
    print("\n===== SAMPLE PREDICTIONS =====")
    for i in range(min(5, len(output_data))):
        print(f"House ID: {output_data[i]['Id']}, "
              f"GrLivArea: {output_data[i]['GrLivArea']:.2f}, "
              f"Predicted SalePrice: ${output_data[i]['SalePrice']:.2f}")

if __name__ == "__main__":
    main()
