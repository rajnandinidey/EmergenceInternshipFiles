import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class RidgeRegression:
    
    #Implementation of Ridge Regression (L2 regularization)
    
    #Ridge regression minimizes the sum of squared errors with an L2 penalty:
    #Loss = (1/n) * ||y - Xw||^2 + alpha * ||w||^2
    
    #where ||w||^2 is the L2 norm (sum of squared values) of the weights
    
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=True, solver='closed_form'):
        
        #Initialize Ridge Regression model
        
        #Parameters:
        #alpha : float, default=1.0
        #(Regularization strength. Higher values mean stronger regularization)
        #fit_intercept : bool, default=True
        #(Whether to calculate the intercept for this model)
        #normalize : bool, default=True
        #(Whether to normalize the input features before fitting)
        #solver : str, default='closed_form'
        #Algorithm to use in the computation:
        #    - 'closed_form': Closed-form solution using normal equation
        #    - 'gradient_descent': Gradient descent optimization
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.solver = solver
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.cost_history_ = []
    
    def fit(self, X, y, learning_rate=0.01, max_iter=1000, tol=1e-4):
        
        #Fit Ridge regression model
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #    Training data
        #y : array-like of shape (n_samples,)
        #    Target values
        #learning_rate : float, default=0.01
        #    Learning rate for gradient descent (only used if solver='gradient_descent')
        #max_iter : int, default=1000
        #    Maximum number of iterations for gradient descent
        #tol : float, default=1e-4
        #    Tolerance for stopping criterion for gradient descent
            
        #Returns:
        #self : object
        #    Returns self
        
        X = np.array(X)
        y = np.array(y)
        
        # Add bias term if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
        else:
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0
        
        # Normalize features
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        n_samples, n_features = X.shape
        
        if self.solver == 'closed_form':
            # Closed-form solution: w = (X^T X + alpha*I)^(-1) X^T y
            identity = np.identity(n_features)
            XTX = X.T @ X
            regularized_matrix = XTX + self.alpha * identity
            self.coef_ = np.linalg.solve(regularized_matrix, X.T @ y)
            
            # Calculate cost for the optimal solution
            y_pred = X @ self.coef_
            mse = np.mean((y - y_pred) ** 2)
            l2_penalty = self.alpha * np.sum(self.coef_ ** 2)
            self.cost_history_.append(mse + l2_penalty)
            
        elif self.solver == 'gradient_descent':
            # Initialize weights
            self.coef_ = np.zeros(n_features)
            
            # Gradient descent
            for i in range(max_iter):
                # Store previous weights for convergence check
                coef_old = self.coef_.copy()
                
                # Compute predictions
                y_pred = X @ self.coef_
                
                # Compute gradients: d/dw = -2X^T(y - Xw)/n + 2*alpha*w
                gradients = (-2/n_samples) * X.T @ (y - y_pred) + 2 * self.alpha * self.coef_
                
                # Update weights
                self.coef_ = self.coef_ - learning_rate * gradients
                
                # Calculate cost
                y_pred = X @ self.coef_
                mse = np.mean((y - y_pred) ** 2)
                l2_penalty = self.alpha * np.sum(self.coef_ ** 2)
                cost = mse + l2_penalty
                self.cost_history_.append(cost)
                
                # Check for convergence
                if np.linalg.norm(self.coef_ - coef_old) < tol:
                    break
        else:
            raise ValueError("Solver must be 'closed_form' or 'gradient_descent'")
        
        # Calculate intercept
        if self.fit_intercept:
            if self.normalize:
                self.intercept_ = y_mean - np.sum(self.coef_ * self.scaler.mean_ / self.scaler.scale_)
            else:
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        
        return self
    
    def predict(self, X):
        
        #Predict using the linear model
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #    Samples
            
        #Returns:
        #y_pred : array-like of shape (n_samples,)
        #    Predicted values
        
        X = np.array(X)
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        
        #Return the coefficient of determination R^2
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #    Test samples
        #y : array-like of shape (n_samples,)
        #    True values
            
        #Returns:
        #score : float
        #    R^2 score
        
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

