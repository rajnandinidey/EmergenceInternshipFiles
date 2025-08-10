import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class Node:
    """
    Node class for Decision Tree
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx    # Index of feature to split on
        self.threshold = threshold        # Threshold value for the split
        self.left = left                  # Left subtree
        self.right = right                # Right subtree
        self.value = value                # Value for leaf node (mean of target values)

class DecisionTreeRegressor:
    """
    Decision Tree Regressor implementation from scratch
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth                # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.min_samples_leaf = min_samples_leaf    # Minimum samples required in a leaf node
        self.root = None                          # Root node of the tree

    def fit(self, X, y):
        """
        Build the decision tree
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf or 
            np.all(y == y[0])):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Find the best split
        best_feature_idx, best_threshold = self._best_split(X, y)
        
        # If no good split is found, create a leaf node
        if best_feature_idx is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices = X[:, best_feature_idx] <= best_threshold
        right_indices = ~left_indices
        
        # Check if split produces valid leaf nodes
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_idx=best_feature_idx, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting
        """
        m = X.shape[0]
        if m <= 1:
            return None, None
        
        # Count of samples in parent node
        parent_var = self._calculate_variance(y)
        
        # If parent variance is 0, no need to split further
        if parent_var == 0:
            return None, None
        
        best_var_reduction = 0.0
        best_feature_idx = None
        best_threshold = None
        
        # Loop through all features
        for feature_idx in range(self.n_features):
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try all possible thresholds
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Skip if split doesn't meet minimum samples requirement
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate variance reduction
                left_var = self._calculate_variance(y[left_indices])
                right_var = self._calculate_variance(y[right_indices])
                
                # Weighted variance of children
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                weighted_var = (n_left * left_var + n_right * right_var) / m
                
                # Calculate variance reduction
                var_reduction = parent_var - weighted_var
                
                # Update best split if this one is better
                if var_reduction > best_var_reduction:
                    best_var_reduction = var_reduction
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def _calculate_variance(self, y):
        """
        Calculate variance of target values
        """
        if len(y) <= 1:
            return 0
        return np.var(y)
    
    def predict(self, X):
        """
        Predict target values for X
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        return np.array([self._predict_sample(sample) for sample in X])
    
    def _predict_sample(self, sample):
        """
        Predict target value for a single sample
        """
        node = self.root
        while node.value is None:  # While not a leaf node
            if sample[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


# Example usage with synthetic data
if __name__ == "__main__":
    # Generate synthetic data for regression
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=X.shape[0])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the decision tree regressor
    dt_regressor = DecisionTreeRegressor(max_depth=5)
    dt_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_regressor.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Sort test data for smooth line plotting
    sort_idx = np.argsort(X_test.ravel())
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Plot the data points
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Testing data')
    
    # Plot the predictions
    plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Predictions')
    
    plt.title('Decision Tree Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
