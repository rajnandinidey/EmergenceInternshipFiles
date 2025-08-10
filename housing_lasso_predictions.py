import csv
import numpy as np
from lasso_regression import LassoRegression

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

def normalize_data(X):
    """Normalize features to prevent numerical issues"""
    X_mean = sum(X) / len(X)
    X_std = (sum((x - X_mean) ** 2 for x in X) / len(X)) ** 0.5
    X_std = max(X_std, 1e-8)  # Prevent division by zero
    
    X_norm = [(x - X_mean) / X_std for x in X]
    
    return X_norm, X_mean, X_std

def main():
    # Load training data
    print("Loading training data...")
    train_file_path = "train_housingprices.csv"
    train_headers, train_data = load_housing_data(train_file_path)
    
    # Load test data
    print("Loading test data...")
    test_file_path = "test_housingprices.csv"
    test_headers, test_data = load_housing_data(test_file_path)
    
    # Select numeric features to use (excluding Id and target)
    numeric_features = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
        'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal', 'YrSold'
    ]
    
    # Find indices of selected features
    feature_indices = []
    for feature in numeric_features:
        if feature in train_headers:
            feature_indices.append(train_headers.index(feature))
    
    # Find index of target (SalePrice) and Id
    target_index = train_headers.index('SalePrice')
    id_index_train = train_headers.index('Id')
    id_index_test = test_headers.index('Id')
    
    # Extract valid rows from training data
    valid_rows = []
    for row in train_data:
        if (target_index < len(row) and row[target_index] is not None and
            all(i < len(row) and row[i] is not None and isinstance(row[i], (int, float)) 
                for i in feature_indices if i < len(row))):
            valid_rows.append(row)
    
    print(f"Loaded {len(valid_rows)} valid rows of training data")
    
    # Extract features and target from training data
    X_train = [[row[i] for i in feature_indices] for row in valid_rows]
    y_train = [row[target_index] for row in valid_rows]
    
    # Calculate mean values for each feature from training data
    feature_means = []
    for i in range(len(feature_indices)):
        feature_values = [row[i] for row in X_train if i < len(row) and row[i] is not None]
        feature_means.append(sum(feature_values) / len(feature_values) if feature_values else 0)
    
    # Process all test data rows and get all test IDs
    all_test_ids = [int(row[id_index_test]) for row in test_data]
    print(f"Total test samples: {len(all_test_ids)}")
    
    # Extract features from test data, handling missing values
    X_test = []
    for row in test_data:
        test_row = []
        for idx, feature_idx in enumerate(feature_indices):
            if feature_idx < len(row) and row[feature_idx] is not None and isinstance(row[feature_idx], (int, float)):
                test_row.append(row[feature_idx])
            else:
                # Use mean from training data for missing values
                test_row.append(feature_means[idx])
        X_test.append(test_row)
    
    # Normalize training data
    print("Normalizing data...")
    X_train_columns = list(zip(*X_train))  # Transpose to get columns
    X_train_norm_columns = []
    X_means = []
    X_stds = []
    
    for col in X_train_columns:
        col_norm, col_mean, col_std = normalize_data(col)
        X_train_norm_columns.append(col_norm)
        X_means.append(col_mean)
        X_stds.append(col_std)
    
    # Transpose back to rows
    X_train_norm = list(zip(*X_train_norm_columns))
    
    # Normalize target
    y_train_norm, y_mean, y_std = normalize_data(y_train)
    
    # Normalize test data using training statistics
    X_test_norm = []
    for row in X_test:
        normalized_row = [(row[i] - X_means[i]) / X_stds[i] for i in range(len(row))]
        X_test_norm.append(normalized_row)
    
    # Try different alpha values for regularization
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
    
    for alpha in alphas:
        print(f"\nTraining Lasso regression model with alpha={alpha}...")
        
        # Create and fit the model
        model = LassoRegression(alpha=alpha, max_iter=5000)
        model.fit(X_train_norm, y_train_norm)
        
        # Make predictions on test data (in normalized space)
        y_pred_norm = model.predict(X_test_norm)
        
        # Denormalize predictions
        y_pred = [pred * y_std + y_mean for pred in y_pred_norm]
        
        # Write predictions to CSV file
        output_file = f"lasso_predictions_alpha_{alpha}.csv"
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Id', 'SalePrice'])
            for i, pred in enumerate(y_pred):
                writer.writerow([all_test_ids[i], round(pred, 4)])
        
        print(f"Predictions saved to {output_file} with {len(y_pred)} predictions")

if __name__ == "__main__":
    main()
