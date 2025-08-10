"""
Polynomial Regression Implementation without using matrices or external libraries
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
        
    def fit(self, x, y):
        """
        Fit polynomial regression model using direct calculation
        
        Parameters:
        x (list): Input features (1D list)
        y (list): Target values
        
        Returns:
        self: Fitted model
        """
        n = len(x)
        degree = self.degree
        
        # Number of coefficients (degree + 1 for the constant term)
        num_coeffs = degree + 1
        
        # Initialize normal equation components
        # We'll build the system of equations directly
        
        # Initialize the normal matrix (corresponds to X^T * X in matrix form)
        normal_matrix = []
        for i in range(num_coeffs):
            row = []
            for j in range(num_coeffs):
                # Calculate sum(x^(i+j))
                power = i + j
                sum_x_power = sum(x_val ** power for x_val in x)
                row.append(sum_x_power)
            normal_matrix.append(row)
        
        # Initialize the right-hand side (corresponds to X^T * y in matrix form)
        rhs = []
        for i in range(num_coeffs):
            # Calculate sum(y * x^i)
            sum_y_x_power = sum(y[j] * (x[j] ** i) for j in range(n))
            rhs.append(sum_y_x_power)
        
        # Solve the system of equations using Gaussian elimination
        self.coefficients = self._gaussian_elimination(normal_matrix, rhs)
        
        return self
    
    def _gaussian_elimination(self, A, b):
        """
        Solve a system of linear equations using Gaussian elimination
        
        Parameters:
        A (list of lists): Coefficient matrix
        b (list): Right-hand side vector
        
        Returns:
        list: Solution vector
        """
        n = len(A)
        
        # Create augmented matrix [A|b]
        augmented = []
        for i in range(n):
            augmented.append(A[i].copy() + [b[i]])
        
        # Forward elimination
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
                raise ValueError("Matrix is singular, cannot solve the system")
            
            # Eliminate entries below pivot
            for j in range(i + 1, n):
                factor = augmented[j][i] / pivot
                for k in range(i, n + 1):
                    augmented[j][k] -= factor * augmented[i][k]
        
        # Back substitution
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i + 1, n):
                x[i] -= augmented[i][j] * x[j]
            x[i] /= augmented[i][i]
            
        return x
    
    def predict(self, x):
        """
        Make predictions using the fitted model
        
        Parameters:
        x (list): Input features
        
        Returns:
        list: Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = []
        for x_val in x:
            # Calculate prediction using Horner's method for polynomial evaluation
            pred = self.coefficients[-1]
            for i in range(len(self.coefficients) - 2, -1, -1):
                pred = pred * x_val + self.coefficients[i]
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
