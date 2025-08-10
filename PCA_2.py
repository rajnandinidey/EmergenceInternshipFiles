import math
import random


class PCA:
    # Principal Component Analysis implementation from scratch.
    
    # Parameters:
    #    n_components (int): Number of principal components to keep
    #    components (list): Principal components (eigenvectors)
    #    explained_variance (list): Variance explained by each component
    #    mean (list): Mean of each feature in the original data
    #    std (list): Standard deviation of each feature in the original data
    
    def __init__(self, n_components=2):
        # Initialize the PCA object with parameters.
        
        # Parameters:
        #    n_components (int): Number of principal components to keep
        
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.mean = None
        self.std = None
    
    def _standardize_data(self, data):
        # Standardize the data (zero mean and unit variance).
        
        # Parameters:
        #    data (list): List of data points
        
        # Returns:
        #    list: Standardized data
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.mean = None
        self.std = None
    
    def _standardize_data(self, data):
        # Standardize the data (zero mean and unit variance).
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    list: Standardized data
        
        n_samples = len(data)
        n_features = len(data[0])
        
        # Calculate mean for each feature
        self.mean = [0] * n_features
        for i in range(n_samples):
            for j in range(n_features):
                self.mean[j] += data[i][j]
        
        for j in range(n_features):
            self.mean[j] /= n_samples
        
        # Calculate standard deviation for each feature
        self.std = [0] * n_features
        for i in range(n_samples):
            for j in range(n_features):
                self.std[j] += (data[i][j] - self.mean[j]) ** 2
        
        for j in range(n_features):
            self.std[j] = math.sqrt(self.std[j] / n_samples)
            # Avoid division by zero
            if self.std[j] == 0:
                self.std[j] = 1
        
        # Standardize the data
        standardized_data = []
        for i in range(n_samples):
            standardized_point = []
            for j in range(n_features):
                standardized_point.append((data[i][j] - self.mean[j]) / self.std[j])
            standardized_data.append(standardized_point)
        
        return standardized_data
    
    def _compute_covariance_matrix(self, data):
        # Compute the covariance matrix of the data.
        
        # Parameters:
        #    data (list): List of standardized data points
            
        # Returns:
        #    list: Covariance matrix
        
        n_samples = len(data)
        n_features = len(data[0])
        
        # Initialize covariance matrix with zeros
        cov_matrix = []
        for i in range(n_features):
            cov_matrix.append([0] * n_features)
        
        # Calculate covariance matrix
        for i in range(n_features):
            for j in range(n_features):
                for k in range(n_samples):
                    cov_matrix[i][j] += data[k][i] * data[k][j]
                cov_matrix[i][j] /= (n_samples - 1)
        
        return cov_matrix
    
    def _power_iteration(self, matrix, num_iterations=100, tolerance=1e-10):
        # Compute the dominant eigenvalue and eigenvector using power iteration.
        
        # Parameters:
        #    matrix (list): Square matrix
        #    num_iterations (int): Maximum number of iterations
        #    tolerance (float): Convergence tolerance
            
        # Returns:
        #    tuple: (eigenvalue, eigenvector)
        
        n = len(matrix)
        
        # Start with a random vector
        vector = [random.random() for _ in range(n)]
        
        # Normalize the vector
        norm = math.sqrt(sum(x**2 for x in vector))
        vector = [x / norm for x in vector]
        
        for _ in range(num_iterations):
            # Matrix-vector multiplication
            new_vector = [0] * n
            for i in range(n):
                for j in range(n):
                    new_vector[i] += matrix[i][j] * vector[j]
            
            # Calculate the norm
            norm = math.sqrt(sum(x**2 for x in new_vector))
            
            # Normalize the vector
            new_vector = [x / norm for x in new_vector]
            
            # Check for convergence
            if all(abs(new_vector[i] - vector[i]) < tolerance for i in range(n)):
                break
            
            vector = new_vector
        
        # Calculate the eigenvalue (Rayleigh quotient)
        eigenvalue = 0
        for i in range(n):
            for j in range(n):
                eigenvalue += vector[i] * matrix[i][j] * vector[j]
        
        return eigenvalue, vector
    
    def _deflate_matrix(self, matrix, eigenvalue, eigenvector):
        # Deflate the matrix by removing the contribution of the found eigenvector.
        
        # Parameters:
        #    matrix (list): Original matrix
        #    eigenvalue (float): Eigenvalue
        #    eigenvector (list): Eigenvector
            
        # Returns:
        #    list: Deflated matrix
        
        n = len(matrix)
        deflated_matrix = []
        
        for i in range(n):
            row = []
            for j in range(n):
                # Subtract the outer product of the eigenvector with itself, scaled by the eigenvalue
                value = matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j]
                row.append(value)
            deflated_matrix.append(row)
        
        return deflated_matrix
    
    def _compute_eigenvectors(self, cov_matrix):
        # Compute the top n_components eigenvalues and eigenvectors.
        
        # Parameters:
        #    cov_matrix (list): Covariance matrix
            
        # Returns:
        #    tuple: (eigenvalues, eigenvectors)
        
        n_features = len(cov_matrix)
        eigenvalues = []
        eigenvectors = []
        
        # Make a copy of the covariance matrix
        matrix = []
        for row in cov_matrix:
            matrix.append(row.copy())
        
        # Find the top n_components eigenvalues and eigenvectors
        for _ in range(min(self.n_components, n_features)):
            eigenvalue, eigenvector = self._power_iteration(matrix)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            
            # Deflate the matrix
            matrix = self._deflate_matrix(matrix, eigenvalue, eigenvector)
        
        # Calculate the total variance (sum of all eigenvalues)
        total_variance = sum(eigenvalues)
        
        # Calculate the explained variance ratio
        self.explained_variance = [eigenvalue / total_variance for eigenvalue in eigenvalues]
        
        return eigenvalues, eigenvectors
    
    def fit(self, data):
        # Fit the PCA model to the data.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    self: The fitted model
        
        # Standardize the data
        standardized_data = self._standardize_data(data)
        
        # Compute the covariance matrix
        cov_matrix = self._compute_covariance_matrix(standardized_data)
        
        # Compute eigenvalues and eigenvectors
        _, eigenvectors = self._compute_eigenvectors(cov_matrix)
        
        # Store the principal components
        self.components = eigenvectors
        
        return self
    
    def transform(self, data):
        # Transform the data using the fitted PCA model.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    list: Transformed data
        
        n_samples = len(data)
        n_features = len(data[0])
        
        # Standardize the data using the stored mean and std
        standardized_data = []
        for i in range(n_samples):
            standardized_point = []
            for j in range(n_features):
                standardized_point.append((data[i][j] - self.mean[j]) / self.std[j])
            standardized_data.append(standardized_point)
        
        # Project the data onto the principal components
        transformed_data = []
        for i in range(n_samples):
            transformed_point = []
            for component in self.components:
                # Dot product of the data point and the component
                value = 0
                for j in range(n_features):
                    value += standardized_data[i][j] * component[j]
                transformed_point.append(value)
            transformed_data.append(transformed_point)
        
        return transformed_data
    
    def fit_transform(self, data):
        # Fit the PCA model to the data and transform it.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    list: Transformed data
        
        self.fit(data)
        return self.transform(data)


# Example usage and visualization
if __name__ == "__main__":
    # Function to generate sample data
    def generate_sample_data(n_samples=100, n_features=3):
        # Generate sample data with correlated features.
        
        # Parameters:
        #    n_samples (int): Number of samples to generate
        #    n_features (int): Number of features
            
        # Returns:
        #    list: Generated data points
        
        data = []
        
        for _ in range(n_samples):
            # Generate a base value
            base = random.uniform(-10, 10)
            
            # Generate correlated features
            point = []
            for j in range(n_features):
                # Each feature is correlated with the base value plus some noise
                value = base * (j + 1) / n_features + random.gauss(0, 0.5)
                point.append(value)
            
            data.append(point)
        
        return data
    
    # Generate sample data
    data = generate_sample_data(n_samples=100, n_features=3)
    
    # Apply PCA
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)
    
    # Print results
    print(f"Number of components: {pca.n_components}")
    print("Explained variance ratio:")
    for i, var in enumerate(pca.explained_variance):
        print(f"  Component {i+1}: {var:.4f} ({var*100:.2f}%)")
    