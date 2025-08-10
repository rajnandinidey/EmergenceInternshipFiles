import numpy as np
import random
from collections import Counter

class Node:
    
    #Node class for Decision Tree
    
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx    # Index of feature to split on
        self.threshold = threshold        # Threshold value for the split
        self.left = left                  # Left subtree
        self.right = right                # Right subtree
        self.value = value                # Value for leaf node (mean of target values)

class DecisionTreeRegressor:
    
    #Decision Tree Regressor implementation from scratch
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth                # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.min_samples_leaf = min_samples_leaf    # Minimum samples required in a leaf node
        self.max_features = max_features          # Maximum number of features to consider for splitting
        self.root = None                          # Root node of the tree
        self.feature_importances_ = None          # Feature importances

    def fit(self, X, y):
        
        #Build the decision tree
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #The training input samples.
        #y : array-like of shape (n_samples,)
        #The target values.
        
        n_samples, self.n_features = X.shape
        
        # Set max_features if not specified
        if self.max_features is None:
            self.max_features = self.n_features
        elif self.max_features == 'sqrt':
            self.max_features = max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == 'log2':
            self.max_features = max(1, int(np.log2(self.n_features)))
        else:
            self.max_features = min(self.max_features, self.n_features)
            
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._grow_tree(X, y)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)
        
    def _grow_tree(self, X, y, depth=0):
        
        #Recursively build the decision tree
        
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
        
        # Update feature importance
        parent_var = self._calculate_variance(y)
        left_var = self._calculate_variance(y[left_indices])
        right_var = self._calculate_variance(y[right_indices])
        n_left, n_right = np.sum(left_indices), np.sum(right_indices)
        weighted_var = (n_left * left_var + n_right * right_var) / n_samples
        var_reduction = parent_var - weighted_var
        
        self.feature_importances_[best_feature_idx] += var_reduction * n_samples
        
        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_idx=best_feature_idx, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _best_split(self, X, y):
        
        #Find the best feature and threshold for splitting
        
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
        
        # Randomly select features to consider for splitting
        feature_indices = np.random.choice(self.n_features, 
                                          size=min(self.max_features, self.n_features), 
                                          replace=False)
        
        # Loop through selected features
        for feature_idx in feature_indices:
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
        
        #Calculate variance of target values
        
        if len(y) <= 1:
            return 0
        return np.var(y)
    
    def predict(self, X):
        
        #Predict target values for X
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #The input samples.
            
        #Returns:
        #y : array-like of shape (n_samples,)
        #The predicted values.
        
        return np.array([self._predict_sample(sample) for sample in X])
    
    def _predict_sample(self, sample):
        
        #Predict target value for a single sample
        
        node = self.root
        while node.value is None:  # While not a leaf node
            if sample[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class RandomForestRegressor:
    
    #Random Forest Regressor implementation from scratch
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, 
                 random_state=None, n_jobs=None):
        self.n_estimators = n_estimators          # Number of trees in the forest
        self.max_depth = max_depth                # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Min samples required to split a node
        self.min_samples_leaf = min_samples_leaf    # Min samples required in a leaf node
        self.max_features = max_features          # Max features to consider for splitting
        self.bootstrap = bootstrap                # Whether to use bootstrap samples
        self.random_state = random_state          # Random state for reproducibility
        self.n_jobs = n_jobs                      # Number of jobs to run in parallel (not used)
        self.trees = []                           # List of decision trees
        self.feature_importances_ = None          # Feature importances
        
    def fit(self, X, y):
        
        #Build a forest of trees from the training data
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #The training input samples.
        #y : array-like of shape (n_samples,)
        #The target values.
        
        n_samples, n_features = X.shape
        
        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        self.trees = []
        
        # Train each tree in the forest
        for i in range(self.n_estimators):
            # Create a decision tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            
            # Create bootstrap sample if bootstrap is True
            if self.bootstrap:
                # Sample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                # Use the entire dataset
                X_sample = X
                y_sample = y
            
            # Train the tree
            tree.fit(X_sample, y_sample)
            
            # Add the tree to the forest
            self.trees.append(tree)
        
        # Calculate feature importances as the average of all trees
        self.feature_importances_ = np.zeros(n_features)
        for tree in self.trees:
            self.feature_importances_ += tree.feature_importances_
        
        # Normalize feature importances
        if len(self.trees) > 0:
            self.feature_importances_ /= len(self.trees)
        
        return self
    
    def predict(self, X):
        
        #Predict regression target for X
        
        #Parameters:
        #X : array-like of shape (n_samples, n_features)
        #The input samples.
            
        #Returns:
        #y : array-like of shape (n_samples,)
        #The predicted values.
        
        #Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Average predictions from all trees
        return np.mean(predictions, axis=0)


#Implementation of train_test_split
def train_test_split(X, y, test_size=0.2, random_state=None):
    
    #Split arrays into random train and test subsets
    
    #Parameters:
    #X : array-like
    #Features data
    #y : array-like
    #Target data
    #test_size : float
    #Proportion of the dataset to include in the test split
    #random_state : int
    #Controls the shuffling applied to the data before applying the split
    
    #Returns:
    #X_train, X_test, y_train, y_test : arrays
    #The train and test subsets
    
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

#Implementation of mean_squared_error
def mean_squared_error(y_true, y_pred):
    
    #Calculate mean squared error between true and predicted values
    
    #Parameters:
    #y_true : array-like
    #True target values
    #y_pred : array-like
    #Predicted target values
    
    #Returns:
    #mse : float
    #Mean squared error
    
    return np.mean((y_true - y_pred) ** 2)

#Implementation of r2_score
def r2_score(y_true, y_pred):
    
    #Calculate R² score (coefficient of determination)
    
    #Parameters:
    #y_true : array-like
    #True target values
    #y_pred : array-like
    #Predicted target values
    
    #Returns:
    #r2 : float
    #R² score
    
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    #Handle edge case where ss_total is 0
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    
    return 1 - (ss_residual / ss_total)

