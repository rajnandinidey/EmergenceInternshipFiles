import numpy as np
import matplotlib.pyplot as plt

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        
        #Initialize Lasso regression model with L1 regularization
        
        #Parameters:
        #alpha (float): Regularization strength (higher values = more regularization)
        #max_iter (int): Maximum number of iterations for coordinate descent
        #tol (float): Convergence tolerance
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        
    def _soft_thresholding(self, x, lambda_):
        
        #Soft thresholding operator for coordinate descent
        
        #Parameters:
        #x (float): Input value
        #lambda_ (float): Threshold value
        
        #Returns:
        #float: Soft thresholded value
        
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        else:
            return 0
    
    def fit(self, X, y):
        
        #Fit Lasso regression model using coordinate descent
        
        #Parameters:
        #X (array-like): Training features of shape (n_samples, n_features)
        #y (array-like): Target values of shape (n_samples,)
        
        #Returns:
        #self: Fitted model
        
        #Convert inputs to numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Standardize features
        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0)
        self.X_std_[self.X_std_ == 0] = 1  # Avoid division by zero
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        # Standardize target
        self.y_mean_ = np.mean(y)
        y_centered = y - self.y_mean_
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        
        # Precompute X^T
        X_T = X_scaled.T
        
        # Precompute squared norms of features
        X_norm_squared = np.sum(X_scaled ** 2, axis=0)
        
        # Coordinate descent
        for iter_idx in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Update each coefficient
            for j in range(n_features):
                if X_norm_squared[j] == 0:
                    continue
                
                # Compute residual without feature j
                y_partial = y_centered - np.dot(X_scaled, self.coef_)
                y_partial += X_scaled[:, j] * self.coef_[j]
                
                # Calculate correlation
                rho = np.dot(X_T[j], y_partial)
                
                # Update coefficient using soft thresholding
                self.coef_[j] = self._soft_thresholding(rho, self.alpha) / X_norm_squared[j]
            
            # Check for convergence
            delta = np.linalg.norm(self.coef_ - coef_old)
            if delta < self.tol:
                break
        
        # Convert coefficients back to original scale
        self.intercept_ = self.y_mean_ - np.sum(self.coef_ * self.X_mean_ / self.X_std_)
        self.coef_ = self.coef_ / self.X_std_
        
        return self
    
    def predict(self, X):
        
        #Making predictions using the fitted model
        
        #Parameters:
        #X (array-like): Input features of shape (n_samples, n_features)
        
        #Returns:
        #array: Predicted values of shape (n_samples,)
        
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X = np.array(X, dtype=float)
        return np.dot(X, self.coef_) + self.intercept_
    
    def score(self, X, y):
        
        #Calculate R^2 score (coefficient of determination)
        
        #Parameters:
        #X (array-like): Input features
        #y (array-like): True target values
        
        #Returns:
        #float: R^2 score
        
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v


# When run directly, demonstrate Lasso regression with synthetic data
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    # Create coefficients with some sparsity (only 5 non-zero)
    true_coef = np.zeros(n_features)
    true_coef[:5] = np.array([3.5, -2.0, 1.5, -1.0, 0.5])
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with noise
    y = np.dot(X, true_coef) + np.random.normal(0, 0.5, n_samples)
    
    # Split data into train and test sets
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Try different alpha values
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
    scores = []
    coefs = []
    
    plt.figure(figsize=(12, 8))
    
    for alpha in alphas:
        # Train model
        lasso = LassoRegression(alpha=alpha)
        lasso.fit(X_train, y_train)
        
        # Evaluate
        score = lasso.score(X_test, y_test)
        scores.append(score)
        coefs.append(lasso.coef_.copy())
        
        print(f"Alpha: {alpha}, R^2 score: {score:.4f}, Non-zero coefficients: {np.sum(np.abs(lasso.coef_) > 1e-10)}")
        
        # Plot coefficients
        plt.plot(range(n_features), lasso.coef_, 'o-', label=f'Alpha = {alpha}')
    
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Coefficients vs Alpha')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('lasso_coefficients.png')
    
    # Plot R^2 score vs alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, scores, 'o-')
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('R^2 Score')
    plt.title('Lasso Performance vs Regularization Strength')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lasso_performance.png')
    
    print("\nTrue coefficients:")
    print(true_coef)
    
    print("\nBest model coefficients (alpha with highest R^2):")
    best_idx = np.argmax(scores)
    print(f"Alpha: {alphas[best_idx]}")
    print(coefs[best_idx])

