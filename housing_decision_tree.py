import numpy as np
import pandas as pd
import sys
import os

# Add the directory containing decision_tree_regression.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the DecisionTreeRegressor from our custom implementation
from decision_tree_regression import DecisionTreeRegressor

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
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 6. Train the Decision Tree Regressor
print("Training Decision Tree Regressor...")
dt_regressor = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2)
dt_regressor.fit(X_train.values, y_train.values)

# 7. Make predictions
print("Making predictions...")
y_pred = dt_regressor.predict(X_test.values)

# 8. Evaluate the model
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
        
    return 1 - (ss_residual / ss_total)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mse = mean_squared_error(y_test.values, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test.values, y_pred)
mae = mean_absolute_error(y_test.values, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# 9. Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Absolute_Error': np.abs(y_test.values - y_pred)
})
predictions_df.to_csv('/root/Rajnandini/test code/dt_housing_predictions.csv', index=False)
print(f"Predictions saved to dt_housing_predictions.csv")
