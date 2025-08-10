import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import our custom models
from lasso_regression import LassoRegression
from ridge_regression import RidgeRegression

# Load the housing data
print("Loading housing data...")
data = pd.read_csv('train_housingprices.csv')

# Basic data exploration
print(f"Dataset shape: {data.shape}")
print(f"Target variable (SalePrice) statistics:")
print(data['SalePrice'].describe())

# Data preprocessing
# Select numerical features only for simplicity
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('Id')  # Remove ID column
numerical_features.remove('SalePrice')  # Remove target variable

print(f"\nUsing {len(numerical_features)} numerical features:")
print(numerical_features[:10], "...")  # Print first 10 features

# Handle missing values by filling with median
X = data[numerical_features].fillna(data[numerical_features].median())
y = data['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso Regression model
print("\n--- Training Lasso Regression Model ---")
alphas_lasso = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
lasso_scores = []

for alpha in alphas_lasso:
    lasso = LassoRegression(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = lasso.predict(X_train_scaled)
    y_pred_test = lasso.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    
    lasso_scores.append((alpha, train_r2, test_r2, test_rmse))
    
    print(f"Lasso (alpha={alpha}):")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: ${test_rmse:.2f}")
    
    # Count non-zero coefficients
    non_zero_coefs = np.sum(np.abs(lasso.coef_) > 1e-10)
    print(f"  Non-zero coefficients: {non_zero_coefs} out of {len(lasso.coef_)}")

# Train Ridge Regression model
print("\n--- Training Ridge Regression Model ---")
alphas_ridge = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_scores = []

for alpha in alphas_ridge:
    # Use closed-form solution for speed
    ridge = RidgeRegression(alpha=alpha, solver='closed_form')
    ridge.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = ridge.predict(X_train_scaled)
    y_pred_test = ridge.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    
    ridge_scores.append((alpha, train_r2, test_r2, test_rmse))
    
    print(f"Ridge (alpha={alpha}):")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: ${test_rmse:.2f}")

# Plot results
plt.figure(figsize=(12, 8))

# Plot R² scores vs alpha
plt.subplot(2, 1, 1)
lasso_alphas, lasso_train_r2, lasso_test_r2, _ = zip(*lasso_scores)
ridge_alphas, ridge_train_r2, ridge_test_r2, _ = zip(*ridge_scores)

plt.semilogx(lasso_alphas, lasso_test_r2, 'o-', label='Lasso Test R²')
plt.semilogx(ridge_alphas, ridge_test_r2, 's-', label='Ridge Test R²')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('R² Score')
plt.title('Test R² Score vs Regularization Strength')
plt.legend()
plt.grid(True)

# Plot RMSE vs alpha
plt.subplot(2, 1, 2)
_, _, _, lasso_rmse = zip(*lasso_scores)
_, _, _, ridge_rmse = zip(*ridge_scores)

plt.semilogx(lasso_alphas, lasso_rmse, 'o-', label='Lasso RMSE')
plt.semilogx(ridge_alphas, ridge_rmse, 's-', label='Ridge RMSE')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('RMSE ($)')
plt.title('Test RMSE vs Regularization Strength')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('housing_model_comparison.png')

# Find best models
best_lasso_idx = np.argmax(lasso_test_r2)
best_ridge_idx = np.argmax(ridge_test_r2)

print("\n--- Best Models ---")
print(f"Best Lasso: alpha={lasso_alphas[best_lasso_idx]}, Test R²={lasso_test_r2[best_lasso_idx]:.4f}, RMSE=${lasso_rmse[best_lasso_idx]:.2f}")
print(f"Best Ridge: alpha={ridge_alphas[best_ridge_idx]}, Test R²={ridge_test_r2[best_ridge_idx]:.4f}, RMSE=${ridge_rmse[best_ridge_idx]:.2f}")

print("\nResults visualization saved to 'housing_model_comparison.png'")
