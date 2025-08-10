import numpy as np
import pandas as pd
import sys
import os
import time

# Add the directory containing random_forest_regression.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the RandomForestRegressor from our custom implementation
from random_forest_regression import RandomForestRegressor, train_test_split, mean_squared_error, r2_score

# Load the housing dataset
data_path = '/root/Rajnandini/test code/train_housingprices.csv'
housing_data = pd.read_csv(data_path)

# Print basic information about the dataset
print(f"Dataset shape: {housing_data.shape}")
print(f"Target variable (SalePrice) statistics:")
print(housing_data['SalePrice'].describe())

# Data preprocessing
# 1. Remove the Id column as it's not a feature
if 'Id' in housing_data.columns:
    housing_data = housing_data.drop('Id', axis=1)

# 2. Handle missing values
# For numerical columns, fill with median
numerical_cols = housing_data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if housing_data[col].isnull().sum() > 0:
        housing_data[col] = housing_data[col].fillna(housing_data[col].median())

# 3. Handle categorical variables
# For categorical columns, convert to one-hot encoding
categorical_cols = housing_data.select_dtypes(include=['object']).columns
housing_data_encoded = pd.get_dummies(housing_data, columns=categorical_cols, drop_first=True)

print(f"Dataset shape after encoding: {housing_data_encoded.shape}")

# 4. Split the data into features and target
X = housing_data_encoded.drop('SalePrice', axis=1)
y = housing_data_encoded['SalePrice']

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 6. Train the Random Forest Regressor
print("Training Random Forest Regressor...")
start_time = time.time()

# Using fewer trees and smaller max_depth for faster training
rf_regressor = RandomForestRegressor(
    n_estimators=20,  # Using fewer trees for faster training
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf_regressor.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# 7. Make predictions
print("Making predictions...")
y_pred = rf_regressor.predict(X_test)

# 8. Evaluate the model
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# 9. Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Absolute_Error': np.abs(y_test - y_pred)
})
predictions_df.to_csv('/root/Rajnandini/test code/rf_housing_predictions.csv', index=False)
print(f"Predictions saved to rf_housing_predictions.csv")

# 10. Feature importance analysis
if hasattr(rf_regressor, 'feature_importances_'):
    # Get feature importances
    feature_importances = rf_regressor.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Display top 20 features
    print("\nTop 20 Most Important Features:")
    print(feature_importance_df.head(20))
    
    # Save feature importances to CSV
    feature_importance_df.to_csv('/root/Rajnandini/test code/rf_feature_importances.csv', index=False)
    print(f"Feature importances saved to rf_feature_importances.csv")
