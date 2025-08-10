import csv
import numpy as np
from linear_regression import predict

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

def main():
    # Model parameters from the training (these are the values we got from our previous run)
    slope = 107.130359
    intercept = 18569.026130
    
    # Load test data
    print("Loading test housing data...")
    test_file_path = "test_housingprices.csv"
    feature_col = 'GrLivArea'
    headers, test_data, ids = load_housing_data(test_file_path, feature_col)
    
    # Find the index of the feature column
    feature_idx = headers.index(feature_col)
    
    # Extract feature values
    X_test = [row[feature_idx] for row in test_data]
    
    # Make predictions
    print("Making predictions...")
    predictions = [slope * x + intercept for x in X_test]
    
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
    output_file = 'housing_price_predictions.csv'
    print(f"Saving predictions to {output_file}...")
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'GrLivArea', 'SalePrice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)
    
    print(f"Predictions saved to {output_file}")
    
    # Also create a submission file with just Id and SalePrice
    submission_file = 'submission.csv'
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
