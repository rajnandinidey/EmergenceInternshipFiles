import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from enhanced_polynomial_regression import EnhancedPolynomialRegression

def load_housing_data(file_path, feature_cols=['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual'], target_col='SalePrice'):
    
    #Load housing data from CSV file
    
    #Parameters:
    #file_path : path to the CSV file
    #feature_cols : list of feature column names
    #target_col : name of the target column (None for test data without target)
    
    #Returns:
    #data : pandas DataFrame containing the data
    #feature_matrix : numpy array of features
    #target_values : numpy array of target values (None for test data without target)
    #ids : list of ID values
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Extract IDs
    ids = data['Id'].values
    
    # Check if all feature columns exist
    for col in feature_cols:
        if col not in data.columns:
            raise ValueError(f"Feature column '{col}' not found in the data")
    
    # Extract features
    feature_matrix = data[feature_cols].values
    
    # Extract target if available
    target_values = None
    if target_col in data.columns:
        target_values = data[target_col].values
    
    return data, feature_matrix, target_values, ids

def normalize_data(X, X_mean=None, X_std=None):
    
    #Normalize features to prevent numerical issues
    
    #Parameters:
    #X : numpy array of feature values
    #X_mean : means for normalization (calculated if None)
    #X_std : standard deviations for normalization (calculated if None)
    
    #Returns:
    #X_norm : normalized features
    #X_mean : means used for normalization
    #X_std : standard deviations used for normalization
    
    if X_mean is None or X_std is None:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.maximum(X_std, 1e-8)  # Prevent division by zero
    
    X_norm = (X - X_mean) / X_std
    
    return X_norm, X_mean, X_std

def denormalize_predictions(y_pred_norm, y_mean, y_std):
    
    #Denormalize predictions back to original scale
    
    #Parameters:
    #y_pred_norm : normalized predictions
    #y_mean : mean of target values
    #y_std : standard deviation of target values
    
    #Returns:
    #y_pred : predictions in original scale
    
    return y_pred_norm * y_std + y_mean

def analyze_feature_importance(model, feature_names):
    
    #Analyze feature importance based on coefficient magnitudes
    
    #Parameters:
    #model : fitted polynomial regression model
    #feature_names : list of feature names
    
    #Returns:
    #importance_df : DataFrame with feature importance information
    
    # Get feature importance
    importance = model.get_feature_importance(feature_names)
    
    # Create DataFrame for easier analysis
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', reverse=True)
    
    return importance_df

def save_model_params(model, X_means, X_stds, y_mean, y_std, feature_names, output_path):
    
    #Save model parameters to a CSV file
    
    #Parameters:
    #model : fitted polynomial regression model
    #X_means : means used for normalization
    #X_stds : standard deviations used for normalization
    #y_mean : mean of target values
    #y_std : standard deviation of target values
    #feature_names : list of feature names
    #output_path : path to save the parameters
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['degree', model.degree])
        writer.writerow(['include_interactions', model.include_interactions])
        writer.writerow(['y_mean', y_mean])
        writer.writerow(['y_std', y_std])
        
        # Write feature names
        for i, feature in enumerate(feature_names):
            writer.writerow([f'feature_name_{i}', feature])
        
        # Write feature means and stds
        for i, (mean, std) in enumerate(zip(X_means, X_stds)):
            writer.writerow([f'X_mean_{i}', mean])
            writer.writerow([f'X_std_{i}', std])
        
        # Write coefficients
        for i, coef in enumerate(model.coefficients):
            writer.writerow([f'coefficient_{i}', coef])
        
        # Generate and write feature names for all polynomial terms
        all_feature_names = model.generate_feature_names(feature_names)
        for i, name in enumerate(all_feature_names):
            if i < len(model.coefficients):
                writer.writerow([f'term_{i}', name])
                writer.writerow([f'coef_{i}', model.coefficients[i]])

def main():
    print("Training Polynomial Regression Model with Interaction Terms")
    
    # Load training data
    train_path = "train_housingprices.csv"
    feature_cols = ['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']
    target_col = 'SalePrice'
    
    print("\nLoading training data...")
    train_data, X_train, y_train, _ = load_housing_data(train_path, feature_cols, target_col)
    print(f"Loaded {len(X_train)} training samples")
    
    # Normalize data
    print("\nNormalizing data...")
    X_norm, X_means, X_stds = normalize_data(X_train)
    y_norm, y_mean, y_std = normalize_data(y_train)
    
    # Train model with interaction terms
    print("\nTraining polynomial regression model with interaction terms...")
    model = EnhancedPolynomialRegression(degree=2, include_interactions=True)
    model.fit(X_norm, y_norm)
    
    # Make predictions on training data
    y_pred_norm = model.predict(X_norm)
    y_pred = denormalize_predictions(y_pred_norm, y_mean, y_std)
    
    # Calculate MSE and R²
    mse = model.mean_squared_error(y_norm, y_pred_norm)
    
    # Calculate R²
    ss_total = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_residual = np.sum((y_train - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    print(f"\nModel performance:")
    print(f"MSE (normalized): {mse:.6f}")
    print(f"R²: {r_squared:.6f}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(model, feature_cols)
    
    # Print top 20 most important features
    print("\nTop 20 most important features:")
    print(importance_df.head(20))
    
    # Save model parameters
    output_path = "polynomial_model_with_interactions_params.csv"
    save_model_params(model, X_means, X_stds, y_mean, y_std, feature_cols, output_path)
    print(f"\nSaved model parameters to {output_path}")
    
    # Load test data
    test_path = "test_housingprices.csv"
    print("\nLoading test data...")
    test_data, X_test, _, test_ids = load_housing_data(test_path, feature_cols, None)
    print(f"Loaded {len(X_test)} test samples")
    
    # Normalize test data
    X_test_norm, _, _ = normalize_data(X_test, X_means, X_stds)
    
    # Make predictions on test data
    print("\nGenerating predictions for test data...")
    test_pred_norm = model.predict(X_test_norm)
    test_pred = denormalize_predictions(test_pred_norm, y_mean, y_std)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_pred
    })
    
    # Save predictions to CSV
    output_path = "housing_polynomial_with_interactions_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")
    
    # Plot actual vs predicted for training data
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_pred, alpha=0.5)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Housing Prices (with Interaction Terms)')
    plt.grid(True)
    plt.savefig('polynomial_with_interactions_predictions.png')
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.tight_layout()
    plt.savefig('polynomial_with_interactions_importance.png')
    
    print("\nTraining and prediction complete!")

if __name__ == "__main__":
    main()
