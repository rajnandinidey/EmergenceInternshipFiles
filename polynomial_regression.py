"""
Polynomial Regression Implementation without using external libraries
"""

class PolynomialRegression:
    def __init__(self, degree=2):
        """
        Initialize polynomial regression model
        
        Parameters:
        degree (int): Degree of the polynomial
        """
        self.degree = degree
        self.coefficients = None
        
    def _generate_polynomial_features(self, X):
        """
        Generate polynomial features from input data
        
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
        
        # Generate polynomial terms up to the specified degree
        for i in range(n_samples):
            for power in range(1, self.degree + 1):
                for j in range(n_features):
                    poly_features[i].append(X[i][j] ** power)
                    
        return poly_features
    
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
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Matrix dimensions do not match for multiplication")
        
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
                    
        return result
    
    def _matrix_inverse(self, matrix):
        """
        Calculate the inverse of a matrix using Gauss-Jordan elimination
        
        Parameters:
        matrix (list of lists): Input matrix
        
        Returns:
        list of lists: Inverse matrix
        """
        n = len(matrix)
        
        # Create augmented matrix [A|I]
        augmented = []
        for i in range(n):
            row = matrix[i].copy()
            for j in range(n):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            augmented.append(row)
        
        # Gauss-Jordan elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for j in range(i + 1, n):
                if abs(augmented[j][i]) > abs(augmented[max_row][i]):
                    max_row = j
            
            # Swap rows if needed
            if max_row != i:
                augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            
            pivot = augmented[i][i]
            if pivot == 0:
                raise ValueError("Matrix is singular and cannot be inverted")
            
            # Scale pivot row
            for j in range(i, 2*n):
                augmented[i][j] /= pivot
            
            # Eliminate other rows
            for j in range(n):
                if j != i:
                    factor = augmented[j][i]
                    for k in range(i, 2*n):
                        augmented[j][k] -= factor * augmented[i][k]
        
        # Extract inverse matrix
        inverse = []
        for i in range(n):
            inverse.append(augmented[i][n:])
            
        return inverse
    
    def fit(self, X, y):
        """
        Fit polynomial regression model
        
        Parameters:
        X (list or list of lists): Training features
        y (list): Target values
        
        Returns:
        self: Fitted model
        """
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Calculate X^T (transpose of X)
        X_T = self._matrix_transpose(X_poly)
        
        # Calculate X^T * X
        X_T_X = self._matrix_multiply(X_T, X_poly)
        
        # Calculate (X^T * X)^-1
        X_T_X_inv = self._matrix_inverse(X_T_X)
        
        # Calculate X^T * y
        y_2d = [[val] for val in y]
        X_T_y = self._matrix_multiply(X_T, y_2d)
        
        # Calculate coefficients = (X^T * X)^-1 * X^T * y
        coef_2d = self._matrix_multiply(X_T_X_inv, X_T_y)
        
        # Convert coefficients to 1D list
        self.coefficients = [coef[0] for coef in coef_2d]
        
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
        
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Make predictions
        predictions = []
        for x in X_poly:
            pred = 0
            for i, coef in enumerate(self.coefficients):
                pred += coef * x[i]
            predictions.append(pred)
            
        return predictions
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate mean squared error
        
        Parameters:
        y_true (list): True target values
        y_pred (list): Predicted values
        
        Returns:
        float: Mean squared error
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred must be the same")
        
        n = len(y_true)
        squared_errors = [(y_true[i] - y_pred[i])**2 for i in range(n)]
        mse = sum(squared_errors) / n
        
        return mse


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    # x = [i for i in range(10)]
    # y = [2 + 3*xi + 1.5*xi**2 + 0.1*xi**3 + 0.5*random() for xi in x]  # cubic function with noise
    
    # Sample data (quadratic relationship)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]  # y = x^2 + 1
    
    # Create and fit the model
    model = PolynomialRegression(degree=2)
    model.fit(x, y)
    
    print("Fitted coefficients:", model.coefficients)
    
    # Make predictions
    y_pred = model.predict(x)
    
    print("Predictions:", y_pred)
    print("Actual values:", y)
    
    # Calculate error
    mse = model.mean_squared_error(y, y_pred)
    print("Mean Squared Error:", mse)
    
    # Test with new data
    x_test = [10, 11, 12]
    y_test_pred = model.predict(x_test)
    print("Predictions for new data:", y_test_pred)
