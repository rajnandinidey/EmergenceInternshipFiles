"""
Enhanced Polynomial Regression Implementation with interaction terms
"""
import numpy as np

class EnhancedPolynomialRegression:
    def __init__(self, degree=2, include_interactions=True):
        """
        Initialize enhanced polynomial regression model
        
        Parameters:
        degree (int): Degree of the polynomial
        include_interactions (bool): Whether to include interaction terms between features
        """
        self.degree = degree
        self.include_interactions = include_interactions
        self.coefficients = None
        self.feature_names = None
        
    def _generate_polynomial_features(self, X):
        """
        Generate polynomial features from input data including interaction terms
        
        Parameters:
        X (list or list of lists): Input features
        
        Returns:
        list of lists: Polynomial features
        """
        # Check if X is a single feature (1D) or multiple features (2D)
        is_1d = not any(isinstance(x, (list, tuple)) for x in X)
        
        if is_1d:
            # Convert 1D to 2D for consistent processing
            X = [[x] for x in X]
            
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize polynomial features with a column of ones (for intercept)
        poly_features = [[1] for _ in range(n_samples)]
        
        # Generate individual feature polynomial terms up to the specified degree
        for i in range(n_samples):
            # First add all the individual feature terms (X₁, X₂, X₃, X₄, X₁², X₂², etc.)
            for power in range(1, self.degree + 1):
                for j in range(n_features):
                    poly_features[i].append(X[i][j] ** power)
            
            # Then add interaction terms if requested
            if self.include_interactions and n_features > 1:
                # Add pairwise interactions (X₁X₂, X₁X₃, etc.)
                for j in range(n_features):
                    for k in range(j + 1, n_features):
                        poly_features[i].append(X[i][j] * X[i][k])
                
                # Add higher-degree interactions if degree > 2
                if self.degree > 2:
                    # Add three-way interactions (X₁X₂X₃, etc.)
                    for j in range(n_features):
                        for k in range(j + 1, n_features):
                            for l in range(k + 1, n_features):
                                poly_features[i].append(X[i][j] * X[i][k] * X[i][l])
                
                # Add degree-2 interactions with individual features if degree > 2
                # (e.g., X₁²X₂, X₁X₂², etc.)
                if self.degree > 2:
                    for j in range(n_features):
                        for k in range(n_features):
                            if j != k:
                                poly_features[i].append((X[i][j] ** 2) * X[i][k])
                    
        return poly_features
    
    def generate_feature_names(self, input_features=None):
        """
        Generate names for all polynomial features
        
        Parameters:
        input_features (list): List of input feature names
        
        Returns:
        list: Names of all polynomial features
        """
        if input_features is None:
            input_features = [f'X{i+1}' for i in range(self._n_features)]
        
        feature_names = ['intercept']
        
        # Individual feature polynomial terms
        for power in range(1, self.degree + 1):
            for j, feat in enumerate(input_features):
                if power == 1:
                    feature_names.append(feat)
                else:
                    feature_names.append(f"{feat}^{power}")
        
        # Interaction terms
        if self.include_interactions and len(input_features) > 1:
            # Pairwise interactions
            for j in range(len(input_features)):
                for k in range(j + 1, len(input_features)):
                    feature_names.append(f"{input_features[j]}*{input_features[k]}")
            
            # Higher-degree interactions if degree > 2
            if self.degree > 2:
                # Three-way interactions
                for j in range(len(input_features)):
                    for k in range(j + 1, len(input_features)):
                        for l in range(k + 1, len(input_features)):
                            feature_names.append(f"{input_features[j]}*{input_features[k]}*{input_features[l]}")
                
                # Degree-2 interactions with individual features
                for j in range(len(input_features)):
                    for k in range(len(input_features)):
                        if j != k:
                            feature_names.append(f"{input_features[j]}^2*{input_features[k]}")
        
        return feature_names
    
    def _matrix_transpose(self, matrix):
        """
        Transpose a matrix
        
        Parameters:
        matrix (list of lists): Input matrix
        
        Returns:
        list of lists: Transposed matrix
        """
        rows = len(matrix)
        cols = len(matrix[0])
        
        transposed = [[0 for _ in range(rows)] for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
                
        return transposed
    
    def _matrix_multiply(self, A, B):
        """
        Multiply two matrices
        
        Parameters:
        A (list of lists): First matrix
        B (list of lists): Second matrix
        
        Returns:
        list of lists: Result of matrix multiplication
        """
        A_rows = len(A)
        A_cols = len(A[0])
        B_rows = len(B)
        B_cols = len(B[0])
        
        if A_cols != B_rows:
            raise ValueError("Matrix dimensions do not match for multiplication")
        
        result = [[0 for _ in range(B_cols)] for _ in range(A_rows)]
        
        for i in range(A_rows):
            for j in range(B_cols):
                for k in range(A_cols):
                    result[i][j] += A[i][k] * B[k][j]
                    
        return result
    
    def _matrix_inverse(self, matrix):
        """
        Calculate the inverse of a matrix using numpy
        
        Parameters:
        matrix (list of lists): Input matrix
        
        Returns:
        list of lists: Inverse of the matrix
        """
        # Convert to numpy array for inverse calculation
        np_matrix = np.array(matrix)
        np_inverse = np.linalg.inv(np_matrix)
        
        # Convert back to list of lists
        inverse = np_inverse.tolist()
        
        return inverse
    
    def fit(self, X, y):
        """
        Fit the polynomial regression model
        
        Parameters:
        X (list or list of lists): Input features
        y (list): Target values
        
        Returns:
        self: Fitted model
        """
        # Check if X is a single feature (1D) or multiple features (2D)
        is_1d = not any(isinstance(x, (list, tuple)) for x in X)
        
        if is_1d:
            # Convert 1D to 2D for consistent processing
            X = [[x] for x in X]
            self._n_features = 1
        else:
            self._n_features = len(X[0])
        
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Convert y to column vector
        if isinstance(y[0], (list, tuple)):
            y = [y_i[0] for y_i in y]
            
        y = [[y_i] for y_i in y]
        
        # Calculate coefficients using normal equation: β = (X^T X)^(-1) X^T y
        X_poly_T = self._matrix_transpose(X_poly)
        X_T_X = self._matrix_multiply(X_poly_T, X_poly)
        X_T_X_inv = self._matrix_inverse(X_T_X)
        X_T_y = self._matrix_multiply(X_poly_T, y)
        
        # Calculate coefficients
        self.coefficients = self._matrix_multiply(X_T_X_inv, X_T_y)
        self.coefficients = [coef[0] for coef in self.coefficients]
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model
        
        Parameters:
        X (list or list of lists): Input features
        
        Returns:
        list: Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet")
        
        # Check if X is a single feature (1D) or multiple features (2D)
        is_1d = not any(isinstance(x, (list, tuple)) for x in X)
        
        if is_1d:
            # Convert 1D to 2D for consistent processing
            X = [[x] for x in X]
            
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Make predictions
        y_pred = []
        for i in range(len(X_poly)):
            pred = 0
            for j in range(len(self.coefficients)):
                pred += X_poly[i][j] * self.coefficients[j]
            y_pred.append(pred)
            
        return y_pred
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate mean squared error
        
        Parameters:
        y_true (list): True values
        y_pred (list): Predicted values
        
        Returns:
        float: Mean squared error
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        n = len(y_true)
        
        # Ensure y_true is a flat list
        if isinstance(y_true[0], (list, tuple)):
            y_true = [y[0] for y in y_true]
            
        # Calculate MSE
        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        
        return mse
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on coefficient magnitudes
        
        Parameters:
        feature_names (list): Names of input features
        
        Returns:
        dict: Feature importance information
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet")
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(self._n_features)]
            
        all_feature_names = self.generate_feature_names(feature_names)
        
        # Create dictionary of feature importance
        importance = {}
        for i, name in enumerate(all_feature_names):
            if i < len(self.coefficients):
                importance[name] = abs(self.coefficients[i])
        
        return importance


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    y = [5, 10, 15, 20, 25]
    
    # Create and fit the model
    model = EnhancedPolynomialRegression(degree=2)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate MSE
    mse = model.mean_squared_error(y, y_pred)
    
    # Print results
    print("Coefficients:", model.coefficients)
    print("MSE:", mse)
    
    # Get feature importance
    feature_names = ['X', 'Y']
    importance = model.get_feature_importance(feature_names)
    
    # Print feature importance
    print("\nFeature Importance:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {imp}")
