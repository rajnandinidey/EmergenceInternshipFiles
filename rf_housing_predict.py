import numpy as np
import pandas as pd
import sys
import os

# Add the directory containing random_forest_regression.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the RandomForestRegressor from our custom implementation
from random_forest_regression import RandomForestRegressor

# Load the training and testing datasets
train_path = '/root/Rajnandini/test code/train_housingprices.csv'
test_path = '/root/Rajnandini/test code/test_housingprices.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Store the test IDs for the final submission
test_ids = test_data['Id'].values

# Data preprocessing for both training and testing data
# 1. Remove the Id column as it's not a feature
train_data = train_data.drop('Id', axis=1)
test_data = test_data.drop('Id', axis=1)

# 2. Handle missing values
# For numerical columns in training data, calculate medians
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
numerical_medians = {}
for col in numerical_cols:
    if col != 'SalePrice':  # Skip the target variable
        numerical_medians[col] = train_data[col].median()
        train_data[col] = train_data[col].fillna(numerical_medians[col])

# Apply the same medians to test data
for col in numerical_cols:
    if col in test_data.columns:  # Make sure the column exists in test data
        test_data[col] = test_data[col].fillna(numerical_medians.get(col, 0))

# 3. Handle categorical variables
# Combine train and test data for consistent one-hot encoding
train_size = train_data.shape[0]
combined_data = pd.concat([train_data.drop('SalePrice', axis=1), test_data])

# For categorical columns, convert to one-hot encoding
categorical_cols = combined_data.select_dtypes(include=['object']).columns
combined_data_encoded = pd.get_dummies(combined_data, columns=categorical_cols, drop_first=True)

# Split back into train and test
X_train_encoded = combined_data_encoded[:train_size]
X_test_encoded = combined_data_encoded[train_size:]

# Get the target variable
y_train = train_data['SalePrice']

print(f"Training features shape after encoding: {X_train_encoded.shape}")
print(f"Testing features shape after encoding: {X_test_encoded.shape}")

# Train the Random Forest Regressor
print("Training Random Forest Regressor...")
rf_regressor = RandomForestRegressor(
    n_estimators=20,  # Using fewer trees for faster training
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf_regressor.fit(X_train_encoded.values, y_train.values)

# Make predictions on the test data
print("Making predictions on test data...")
test_predictions = rf_regressor.predict(X_test_encoded.values)

# Create submission dataframe
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

# Save predictions to CSV file
output_path = '/root/Rajnandini/test code/rf_housing_submission.csv'
submission.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

# Display the first few predictions
print("\nFirst 5 predictions:")
print(submission.head())
