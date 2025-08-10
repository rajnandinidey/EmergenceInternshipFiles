import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ridge_regression import RidgeRegression

# Load the training data to fit the scaler
print("Loading training data...")
train_data = pd.read_csv('train_housingprices.csv')

# Select numerical features only for simplicity
numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('Id')  # Remove ID column
numerical_features.remove('SalePrice')  # Remove target variable

print(f"Using {len(numerical_features)} numerical features")

# Handle missing values by filling with median
X_train = train_data[numerical_features].fillna(train_data[numerical_features].median())
y_train = train_data['SalePrice']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the Ridge model with the best alpha from previous experiment
print("Training Ridge regression model...")
best_alpha = 10.0  # Using the best alpha from previous experiment
ridge = RidgeRegression(alpha=best_alpha, solver='closed_form')
ridge.fit(X_train_scaled, y_train)

# Load the test data
print("Loading test data...")
test_data = pd.read_csv('test_housingprices.csv')

# Extract IDs
test_ids = test_data['Id'].values

# Prepare test features (same preprocessing as training data)
X_test = test_data[numerical_features].fillna(train_data[numerical_features].median())
X_test_scaled = scaler.transform(X_test)

# Make predictions
print("Making predictions...")
predictions = ridge.predict(X_test_scaled)

# Round predictions to 2 decimal places
predictions = np.round(predictions, 2)

# Create output DataFrame
output = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

# Save to CSV
output_file = 'ridge_predictions.csv'
output.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
print(f"Total predictions: {len(predictions)}")
print("Sample predictions:")
print(output.head())
