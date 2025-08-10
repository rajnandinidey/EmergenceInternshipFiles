import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class RidgeRegression:
    """
    Implementation of Ridge Regression (L2 regularization) for housing data
    
    Ridge regression minimizes the sum of squared errors with an L2 penalty:
    Loss = (1/n) * ||y - Xw||^2 + alpha * ||w||^2
    
    where ||w||^2 is the L2 norm (sum of squared values) of the weights
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, learning_rate=0.01):
        """
        Initialize Ridge Regression model
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength. Higher values mean stronger regularization.
        max_iter : int, default=1000
            Maximum number of iterations for gradient descent
        tol : float, default=1e-4
            Tolerance for stopping criterion
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Fit Ridge regression model using gradient descent
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Store previous weights for convergence check
            weights_old = self.weights.copy()
            
            # Compute predictions
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute gradients
            dw = (-2/n_samples) * X.T.dot(y - y_pred) + 2 * self.alpha * self.weights
            db = (-2/n_samples) * np.sum(y - y_pred)
            
            # Update weights and bias
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
            # Calculate cost
            y_pred = X.dot(self.weights) + self.bias
            mse = np.mean((y - y_pred) ** 2)
            l2_penalty = self.alpha * np.sum(self.weights ** 2)
            cost = mse + l2_penalty
            self.cost_history.append(cost)
            
            # Check for convergence
            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            R^2 score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def plot_cost_history(self):
        """
        Plot the cost history during training
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.grid(True)
        plt.show()


def load_housing_data(train_path, test_path=None):
    """
    Load and preprocess housing data
    
    Parameters:
    -----------
    train_path : str
        Path to the training data CSV file
    test_path : str, optional
        Path to the test data CSV file
    
    Returns:
    --------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like, optional
        Test features (if test_path is provided)
    y_test : array-like, optional
        Test target values (if test_path is provided)
    feature_names : list
        Names of features used in the model
    """
    # Load training data
    train_data = pd.read_csv(train_path)
    
    # Extract target variable (SalePrice)
    y_train = train_data['SalePrice'].values
    
    # Select numerical features
    numerical_features = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
        'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
        'PoolArea', 'MiscVal'
    ]
    
    # Select categorical features
    categorical_features = [
        'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 
        'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 
        'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 
        'CentralAir', 'Electrical', 'GarageType', 'SaleType', 
        'SaleCondition'
    ]
    
    # Create feature matrix for training data
    X_train_num = train_data[numerical_features].copy()
    X_train_cat = train_data[categorical_features].copy()
    
    # Handle missing values
    X_train_num.fillna(X_train_num.mean(), inplace=True)
    X_train_cat.fillna('Missing', inplace=True)
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_encoded = encoder.fit_transform(X_train_cat)
    
    # Combine numerical and encoded categorical features
    X_train = np.hstack([X_train_num.values, X_train_cat_encoded])
    
    # Create feature names
    numerical_feature_names = numerical_features
    categorical_feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = encoder.categories_[i]
        for category in categories:
            categorical_feature_names.append(f"{feature}_{category}")
    
    feature_names = numerical_feature_names + categorical_feature_names
    
    # If test path is provided, process test data
    if test_path:
        test_data = pd.read_csv(test_path)
        y_test = test_data['SalePrice'].values
        
        X_test_num = test_data[numerical_features].copy()
        X_test_cat = test_data[categorical_features].copy()
        
        X_test_num.fillna(X_train_num.mean(), inplace=True)
        X_test_cat.fillna('Missing', inplace=True)
        
        X_test_cat_encoded = encoder.transform(X_test_cat)
        X_test = np.hstack([X_test_num.values, X_test_cat_encoded])
        
        return X_train, y_train, X_test, y_test, feature_names
    
    return X_train, y_train, feature_names


def find_optimal_alpha(X, y, alphas, n_folds=5):
    """
    Find the optimal alpha value using cross-validation
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    alphas : list
        List of alpha values to try
    n_folds : int, default=5
        Number of folds for cross-validation
    
    Returns:
    --------
    best_alpha : float
        Alpha value with the best cross-validation score
    cv_scores : dict
        Dictionary with alpha values as keys and mean CV scores as values
    """
    cv_scores = {}
    
    for alpha in alphas:
        scores = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)
            
            # Fit model
            model = RidgeRegression(alpha=alpha)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate model
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)
        
        cv_scores[alpha] = np.mean(scores)
    
    best_alpha = max(cv_scores, key=cv_scores.get)
    return best_alpha, cv_scores


def plot_feature_importance(feature_names, weights, top_n=20):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    weights : array-like
        Model weights
    top_n : int, default=20
        Number of top features to display
    """
    # Create a DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(weights)
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Absolute Weight')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load housing data
    train_path = '/root/Rajnandini/test code/train_housingprices.csv'
    test_path = '/root/Rajnandini/test code/test_housingprices.csv'
    
    try:
        X_train, y_train, X_test, y_test, feature_names = load_housing_data(train_path, test_path)
        print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
        print(f"Number of features: {X_train.shape[1]}")
    except:
        # If test data is not available or there's an issue with loading both
        X_train, y_train, feature_names = load_housing_data(train_path)
        print(f"Loaded {X_train.shape[0]} training samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Create a train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal alpha
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha, cv_scores = find_optimal_alpha(X_train, y_train, alphas)
    
    print(f"\nCross-validation results:")
    for alpha, score in cv_scores.items():
        print(f"Alpha = {alpha}: R² = {score:.4f}")
    
    print(f"\nBest alpha: {best_alpha}")
    
    # Train model with best alpha
    ridge = RidgeRegression(alpha=best_alpha, max_iter=5000, learning_rate=0.01)
    ridge.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = ridge.score(X_train_scaled, y_train)
    test_score = ridge.score(X_test_scaled, y_test)
    
    y_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nModel evaluation:")
    print(f"Training R²: {train_score:.4f}")
    print(f"Testing R²: {test_score:.4f}")
    print(f"RMSE: ${rmse:.2f}")
    
    # Plot cost history
    ridge.plot_cost_history()
    
    # Plot feature importance
    plot_feature_importance(feature_names, ridge.weights)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Housing Prices')
    plt.grid(True)
    plt.show()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()
