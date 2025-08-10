import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score
from lasso_regression import LassoRegression
from ridge_regression import RidgeRegression
from sklearn.preprocessing import PolynomialFeatures
from decision_tree_regression import DecisionTreeRegressor

def load_housing_data(file_path):
    """Load housing data from CSV file into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_skewness(data):
    """Calculate skewness of the data"""
    return stats.skew(data)

def apply_optimized_transformations(train_df, test_df):
    """Apply optimized transformations to the datasets"""
    print("Applying optimized transformations...")
    
    # Make copies to avoid modifying the originals
    train_transformed = train_df.copy()
    test_transformed = test_df.copy()
    
    # Features with very high percentage of zeros (>90%) - convert to binary
    high_zero_features = [
        'PoolArea', '3SsnPorch', 'LowQualFinSF', 'MiscVal', 
        'BsmtHalfBath', 'ScreenPorch'
    ]
    
    # Features with moderate to high percentage of zeros (10-90%) - keep as is
    moderate_zero_features = [
        'BsmtFinSF2', 'EnclosedPorch', 'HalfBath', 'MasVnrArea',
        'BsmtFullBath', '2ndFlrSF', 'WoodDeckSF', 'Fireplaces',
        'OpenPorchSF', 'BsmtFinSF1'
    ]
    
    # Features with low percentage of zeros (<10%) - apply log1p
    low_zero_features = [
        'BsmtUnfSF', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
        'FullBath', 'BedroomAbvGr', 'KitchenAbvGr'
    ]
    
    # Other numeric features that don't have zeros but might benefit from log1p
    other_numeric_features = [
        'LotArea', 'GrLivArea', '1stFlrSF', 'LotFrontage',
        'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd'
    ]
    
    # Apply binary encoding to high zero features
    print("\nApplying binary encoding to features with high percentage of zeros:")
    for feature in high_zero_features:
        if feature in train_df.columns:
            train_transformed[f"{feature}_binary"] = (train_df[feature] > 0).astype(int)
            print(f"  - Created {feature}_binary")
        
        if feature in test_df.columns:
            test_transformed[f"{feature}_binary"] = (test_df[feature] > 0).astype(int)
    
    # Apply log1p to low zero features and other numeric features
    log1p_features = low_zero_features + other_numeric_features
    print("\nApplying log1p transformation to features with low percentage of zeros:")
    for feature in log1p_features:
        if feature in train_df.columns:
            # Check if feature contains valid numeric data
            if train_df[feature].dtype in ['int64', 'float64']:
                # Original skewness
                original_skewness = calculate_skewness(train_df[feature].dropna())
                
                # Apply log1p transformation
                train_transformed[f"{feature}_log1p"] = np.log1p(train_df[feature])
                
                # New skewness
                new_skewness = calculate_skewness(train_transformed[f"{feature}_log1p"].dropna())
                
                print(f"  - {feature}: Skewness improved from {original_skewness:.4f} to {new_skewness:.4f}")
        
        if feature in test_df.columns:
            if test_df[feature].dtype in ['int64', 'float64']:
                test_transformed[f"{feature}_log1p"] = np.log1p(test_df[feature])
    
    # Handle SalePrice separately for training data
    if 'SalePrice' in train_df.columns:
        original_skewness = calculate_skewness(train_df['SalePrice'].dropna())
        train_transformed['SalePrice_log1p'] = np.log1p(train_df['SalePrice'])
        new_skewness = calculate_skewness(train_transformed['SalePrice_log1p'].dropna())
        print(f"\nTarget variable SalePrice: Skewness improved from {original_skewness:.4f} to {new_skewness:.4f}")
    
    return train_transformed, test_transformed

def prepare_model_data(train_df, test_df):
    """Prepare data for modeling by selecting appropriate features"""
    # Features to use for modeling
    binary_features = [col for col in train_df.columns if col.endswith('_binary')]
    log1p_features = [col for col in train_df.columns if col.endswith('_log1p') and col != 'SalePrice_log1p']
    
    # Original moderate zero features to keep
    moderate_features = [
        'BsmtFinSF2', 'EnclosedPorch', 'HalfBath', 'MasVnrArea',
        'BsmtFullBath', '2ndFlrSF', 'WoodDeckSF', 'Fireplaces',
        'OpenPorchSF', 'BsmtFinSF1'
    ]
    
    # Other potentially useful features
    other_features = [
        'OverallQual', 'OverallCond', 'MSSubClass'
    ]
    
    # Combine all feature lists
    all_features = binary_features + log1p_features + moderate_features + other_features
    
    # Filter to only include columns that exist in both dataframes
    model_features = [f for f in all_features if f in train_df.columns and f in test_df.columns]
    
    print(f"\nSelected {len(model_features)} features for modeling")
    
    # Extract features and target
    X_train = train_df[model_features].fillna(0).values
    y_train = train_df['SalePrice_log1p'].values if 'SalePrice_log1p' in train_df.columns else train_df['SalePrice'].values
    X_test = test_df[model_features].fillna(0).values
    
    return X_train, y_train, X_test, model_features

def train_models(X_train, y_train):
    """Train multiple regression models"""
    models = {}
    
    # Lasso models
    lasso_alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    for alpha in lasso_alphas:
        print(f"Training Lasso model with alpha={alpha}...")
        model = LassoRegression(alpha=alpha, max_iter=10000, tol=1e-6)
        model.fit(X_train, y_train)
        models[f"lasso_{alpha}"] = model
    
    # Ridge models
    ridge_alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    for alpha in ridge_alphas:
        print(f"Training Ridge model with alpha={alpha}...")
        model = RidgeRegression(alpha=alpha, solver='closed_form')
        model.fit(X_train, y_train)
        models[f"ridge_{alpha}"] = model
    
    # Polynomial models with Ridge regularization
    poly_degrees = [1, 2]
    for degree in poly_degrees:
        print(f"Training Polynomial model with degree={degree}...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_train)
        
        model = RidgeRegression(alpha=0.1, solver='closed_form')
        model.fit(X_poly, y_train)
        
        models[f"poly_{degree}"] = {
            'model': model,
            'poly': poly
        }
    
    # Decision Tree models
    tree_depths = [5, 10, 15]
    for depth in tree_depths:
        print(f"Training Decision Tree model with max_depth={depth}...")
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X_train, y_train)
        models[f"tree_{depth}"] = model
    
    return models

def generate_predictions(models, X_test, test_ids):
    """Generate predictions from all models"""
    predictions = {}
    prediction_files = []
    
    for name, model in models.items():
        print(f"Generating predictions for {name}...")
        
        # Make predictions
        if name.startswith('poly'):
            poly = model['poly']
            X_test_poly = poly.transform(X_test)
            y_pred = model['model'].predict(X_test_poly)
        else:
            y_pred = model.predict(X_test)
        
        # Store predictions
        predictions[name] = y_pred
        
        # Save predictions to file
        output_file = f"optimized_{name}_predictions.csv"
        with open(output_file, 'w', newline='') as file:
            import csv
            writer = csv.writer(file)
            writer.writerow(['Id', 'SalePrice'])
            
            for i, pred in enumerate(y_pred):
                # Transform prediction back to original scale
                original_pred = np.expm1(pred)  # inverse of log1p
                writer.writerow([test_ids[i], round(original_pred, 4)])
        
        print(f"  Saved to {output_file}")
        prediction_files.append(output_file)
    
    return predictions, prediction_files

def create_stacked_predictions(prediction_files, test_ids):
    """Create stacked predictions from multiple models"""
    all_predictions = {}
    
    # Load predictions from each file
    for file_path in prediction_files:
        with open(file_path, 'r') as file:
            import csv
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if len(row) >= 2:
                    id_val = int(row[0])
                    pred = float(row[1])
                    
                    if id_val not in all_predictions:
                        all_predictions[id_val] = []
                    all_predictions[id_val].append(pred)
    
    # Calculate average predictions
    final_predictions = {}
    for id_val, preds in all_predictions.items():
        if preds:
            final_predictions[id_val] = sum(preds) / len(preds)
    
    # Save stacked predictions
    output_file = "optimized_stacked_predictions.csv"
    with open(output_file, 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        
        for id_val in sorted(final_predictions.keys()):
            pred = final_predictions[id_val]
            writer.writerow([id_val, round(pred, 4)])
    
    print(f"Stacked predictions saved to {output_file}")
    return output_file

def main():
    print("Optimizing housing data transformations and generating predictions...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists('optimized_data'):
        os.makedirs('optimized_data')
    
    # Load data
    train_file_path = "train_housingprices.csv"
    test_file_path = "test_housingprices.csv"
    
    train_df = load_housing_data(train_file_path)
    test_df = load_housing_data(test_file_path)
    
    if train_df is None or test_df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded training data with {train_df.shape[0]} rows and {train_df.shape[1]} columns")
    print(f"Loaded test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns")
    
    # Apply optimized transformations
    train_transformed, test_transformed = apply_optimized_transformations(train_df, test_df)
    
    # Save transformed data
    train_transformed.to_csv('optimized_data/train_optimized.csv', index=False)
    test_transformed.to_csv('optimized_data/test_optimized.csv', index=False)
    print("\nSaved transformed data to optimized_data directory")
    
    # Prepare data for modeling
    X_train, y_train, X_test, model_features = prepare_model_data(train_transformed, test_transformed)
    
    # Get test IDs
    test_ids = test_df['Id'].values
    
    # Train models
    print("\n--- Training Models ---")
    models = train_models(X_train, y_train)
    
    # Generate predictions
    print("\n--- Generating Predictions ---")
    predictions, prediction_files = generate_predictions(models, X_test, test_ids)
    
    # Create stacked predictions
    print("\n--- Creating Stacked Predictions ---")
    stacked_file = create_stacked_predictions(prediction_files, test_ids)
    
    print("\nOptimized transformation and prediction process complete!")

if __name__ == "__main__":
    main()
