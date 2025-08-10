import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LassoRegression:
    """
    Implementation of Lasso Regression using proximal gradient descent.
    
    Lasso regression minimizes:
        L(w) = (1/2n) * ||y - Xw||^2 + alpha * ||w||_1
    
    where ||w||_1 is the L1 norm (sum of absolute values) of the weights.
    """
    
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4, learning_rate=0.01):
        """
        Initialize Lasso Regression model
        
        Parameters:
        -----------
        alpha : float, default=0.1
            Regularization parameter that controls the strength of the L1 penalty
        max_iter : int, default=1000
            Maximum number of iterations for the optimization algorithm
        tol : float, default=1e-4
            Tolerance for the stopping criterion
        learning_rate : float, default=0.01
            Step size for the proximal gradient descent
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = 0.0
        self.cost_history_ = []
        self.coef_path_ = []
    
    def _soft_thresholding(self, x, lambda_):
        """
        Soft thresholding operator for proximal gradient descent
        
        S_λ(x) = sign(x) * max(|x| - λ, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    def fit(self, X, y, standardize=True):
        """
        Fit Lasso regression model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        standardize : bool, default=True
            Whether to standardize the data before fitting
            
        Returns:
        --------
        self : object
            Returns self
        """
        X = np.array(X)
        y = np.array(y)
        
        # Store original data for prediction
        self.X_mean_ = np.mean(X, axis=0) if standardize else np.zeros(X.shape[1])
        self.X_std_ = np.std(X, axis=0) if standardize else np.ones(X.shape[1])
        self.y_mean_ = np.mean(y) if standardize else 0.0
        
        # Standardize data
        if standardize:
            X = (X - self.X_mean_) / self.X_std_
            y = y - self.y_mean_
        
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        self.coef_path_.append(self.coef_.copy())
        
        # Precompute X^T * X and X^T * y for efficiency
        XTX = X.T @ X / n_samples
        XTy = X.T @ y / n_samples
        
        # Compute Lipschitz constant for step size
        lipschitz = np.linalg.norm(XTX, 2)
        step_size = 1.0 / lipschitz if lipschitz > 0 else self.learning_rate
        
        # Proximal gradient descent
        for i in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Gradient descent step on the smooth part (MSE)
            gradient = XTX @ self.coef_ - XTy
            self.coef_ = self.coef_ - step_size * gradient
            
            # Proximal operator step on the non-smooth part (L1)
            self.coef_ = self._soft_thresholding(self.coef_, self.alpha * step_size)
            
            # Store coefficients path
            self.coef_path_.append(self.coef_.copy())
            
            # Compute cost
            y_pred = X @ self.coef_
            mse = np.mean((y - y_pred) ** 2)
            l1_penalty = self.alpha * np.sum(np.abs(self.coef_))
            cost = mse + l1_penalty
            self.cost_history_.append(cost)
            
            # Check convergence
            if np.linalg.norm(self.coef_ - coef_old) < self.tol:
                break
        
        # Set intercept
        if standardize:
            # Adjust intercept for standardized data
            self.intercept_ = self.y_mean_ - np.sum(self.coef_ * self.X_mean_ / self.X_std_)
        else:
            self.intercept_ = np.mean(y - X @ self.coef_)
        
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
        
        # Apply same transformation as during fit
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        return X_scaled @ self.coef_ + self.intercept_
    
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
    
    def plot_convergence(self):
        """Plot the convergence of the cost function"""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.cost_history_)), self.cost_history_)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function Convergence')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        coef_path = np.array(self.coef_path_)
        for i in range(min(5, coef_path.shape[1])):  # Plot first 5 coefficients
            plt.plot(range(len(self.coef_path_)), coef_path[:, i], 
                     label=f'Coef {i}')
        plt.xlabel('Iterations')
        plt.ylabel('Coefficient Value')
        plt.title('Coefficient Path')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_coefficients(self, feature_names=None):
        """Plot the coefficients to visualize feature importance"""
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(len(self.coef_))]
        
        # Sort coefficients by absolute value
        sorted_idx = np.argsort(np.abs(self.coef_))[::-1]
        sorted_coef = self.coef_[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        # Plot only non-zero coefficients
        non_zero = np.abs(sorted_coef) > 1e-10
        if np.sum(non_zero) == 0:
            print("All coefficients are zero. Try decreasing alpha.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(np.sum(non_zero)), sorted_coef[non_zero])
        plt.xticks(range(np.sum(non_zero)), [sorted_names[i] for i, nz in enumerate(non_zero) if nz], 
                  rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Lasso Coefficients (Non-zero only)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


# Example usage with synthetic data
if __name__ == "__main__":
    # Create synthetic dataset with sparse coefficients
    np.random.seed(42)
    n_samples, n_features = 200, 50
    
    # Create sparse coefficient vector (only 5 features are relevant)
    true_coef = np.zeros(n_features)
    true_coef[:5] = [1.5, -2.0, 0.8, -1.0, 1.2]
    
    # Generate features matrix with correlated features
    X = np.random.randn(n_samples, n_features)
    # Add correlation between first 10 features
    X[:, 5:10] = X[:, :5] * 0.5 + np.random.randn(n_samples, 5) * 0.5
    
    # Generate target with noise
    y = X @ true_coef + np.random.normal(0, 0.5, size=n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Try different alpha values
    alphas = [0.001, 0.01, 0.1, 1.0]
    plt.figure(figsize=(15, 10))
    
    for i, alpha in enumerate(alphas):
        # Fit model
        lasso = LassoRegression(alpha=alpha, max_iter=2000, tol=1e-6)
        lasso.fit(X_train, y_train)
        
        # Evaluate model
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)
        mse = mean_squared_error(y_test, lasso.predict(X_test))
        
        # Count non-zero coefficients
        n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-10)
        
        print(f"\nLasso Regression (alpha={alpha}):")
        print(f"  Non-zero coefficients: {n_nonzero}/{n_features}")
        print(f"  Training R²: {train_score:.4f}")
        print(f"  Test R²: {test_score:.4f}")
        print(f"  Test MSE: {mse:.4f}")
        
        # Plot coefficients
        plt.subplot(2, 2, i+1)
        plt.stem(range(n_features), lasso.coef_, markerfmt='ro', linefmt='r-', basefmt='b-')
        plt.stem(range(n_features), true_coef, markerfmt='go', linefmt='g-', basefmt='b-')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title(f'Alpha={alpha}, Non-zero: {n_nonzero}, Test R²: {test_score:.4f}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Show convergence for the best model
    best_alpha = 0.1  # Choose based on results
    lasso = LassoRegression(alpha=best_alpha, max_iter=2000, tol=1e-6)
    lasso.fit(X_train, y_train)
    lasso.plot_convergence()
    
    # Compare predictions with true values
    y_pred = lasso.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True vs Predicted Values (alpha={best_alpha})')
    plt.grid(True)
    plt.show()
