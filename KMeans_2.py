import random
import math


class KMeans:
    #Attributes:
    #    n_clusters (int): Number of clusters to form
    #    max_iter (int): Maximum number of iterations for the algorithm
    #    tol (float): Tolerance for centroid convergence
    #    centroids (list): List of cluster centroids
    #    labels (list): Cluster assignments for each data point
    
    def __init__(self, n_clusters=3, max_iter=100, tol=0.0001):
        #Initialize the KMeans object with parameters.
        
        # Parameters:
        #    n_clusters (int): Number of clusters to form
        #    max_iter (int): Maximum number of iterations for the algorithm
        #    tol (float): Tolerance for centroid convergence
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def _euclidean_distance(self, point1, point2):
        #Calculate the Euclidean distance between two points.
        
        # Parameters:
        #    point1 (list): First point coordinates
        #    point2 (list): Second point coordinates
            
        # Returns:
        #    float: Euclidean distance between the points
        
        #Sum of squared differences
        squared_diff_sum = 0
        for i in range(len(point1)):
            squared_diff_sum += (point1[i] - point2[i]) ** 2
        
        # Return the square root of the sum
        return math.sqrt(squared_diff_sum)
    
    def _initialize_centroids(self, data):
        #Initialize cluster centroids by randomly selecting points from the dataset.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    list: Initial centroids
        
        #Create a copy of the data to avoid modifying the original
        data_copy = data.copy()
        #Shuffle the data
        random.shuffle(data_copy)
        # Select the first n_clusters points as initial centroids
        return data_copy[:self.n_clusters]
    
    def _assign_clusters(self, data, centroids):
        #Assign each data point to the nearest centroid.
        
        # Parameters:
        #    data (list): List of data points
        #    centroids (list): List of centroid points
            
        # Returns:
        #    list: Cluster assignments for each data point
        
        clusters = []
        
        for point in data:
            # Calculate distances to all centroids
            distances = [self._euclidean_distance(point, centroid) for centroid in centroids]
            # Assign to the closest centroid (minimum distance)
            cluster_idx = distances.index(min(distances))
            clusters.append(cluster_idx)
            
        return clusters
    
    def _update_centroids(self, data, clusters):
        #Update centroids based on the mean of points in each cluster.
        
        # Parameters:
        #    data (list): List of data points
        #    clusters (list): Cluster assignments for each data point
            
        # Returns:
        #    list: Updated centroids
        
        # Initialize new centroids
        n_features = len(data[0])
        new_centroids = [[0] * n_features for _ in range(self.n_clusters)]
        counts = [0] * self.n_clusters
        
        # Sum up all points in each cluster
        for idx, cluster in enumerate(clusters):
            counts[cluster] += 1
            for j in range(n_features):
                new_centroids[cluster][j] += data[idx][j]
        
        # Calculate the mean for each cluster
        for i in range(self.n_clusters):
            # Avoid division by zero if a cluster is empty
            if counts[i] > 0:
                for j in range(n_features):
                    new_centroids[i][j] /= counts[i]
        
        return new_centroids
    
    def _has_converged(self, old_centroids, new_centroids):
        #Check if the algorithm has converged based on centroid movement.
        
        # Parameters:
        #    old_centroids (list): Previous centroids
        #    new_centroids (list): Updated centroids
            
        # Returns:
        #    bool: True if converged, False otherwise
        
        # Calculate the sum of distances between old and new centroids
        total_distance = 0
        for i in range(len(old_centroids)):
            total_distance += self._euclidean_distance(old_centroids[i], new_centroids[i])
        
        # If the total movement is less than tolerance, consider it converged
        return total_distance < self.tol
    
    def fit(self, data):
        # Fit the K-Means model to the data.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    self: The fitted model
        
        #Initialize centroids
        self.centroids = self._initialize_centroids(data)
        
        # Iterate until convergence or max iterations
        for _ in range(self.max_iter):
            # Assign clusters
            self.labels = self._assign_clusters(data, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(data, self.labels)
            
            # Check for convergence
            if self._has_converged(self.centroids, new_centroids):
                break
                
            # Update centroids for next iteration
            self.centroids = new_centroids
        
        return self
    
    def predict(self, data):
        # Predict the closest cluster for each sample in the data.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    list: Predicted cluster index for each point
        
        return self._assign_clusters(data, self.centroids)


# Example usage and visualization
if __name__ == "__main__":
    # Generate some random 2D data
    def generate_random_data(n_samples=300, n_centers=3, std=1.0):
        # Generate random clustered data
        
        # Parameters:
        #    n_samples (int): Number of samples to generate
        #    n_centers (int): Number of cluster centers
        #    std (float): Standard deviation of the clusters
        
        # Returns:
        #    tuple: (data, true_labels)
        
        centers = []
        for _ in range(n_centers):
            centers.append([random.uniform(-10, 10), random.uniform(-10, 10)])
        
        data = []
        true_labels = []
        
        for i in range(n_samples):
            # Choose a random center
            center_idx = random.randint(0, n_centers-1)
            center = centers[center_idx]
            
            # Generate a point around that center
            point = [
                center[0] + random.gauss(0, std),
                center[1] + random.gauss(0, std)
            ]
            
            data.append(point)
            true_labels.append(center_idx)
            
        return data, true_labels
    
    # Generate data
    data, true_labels = generate_random_data(n_samples=300, n_centers=3, std=1.0)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=3, max_iter=100)
    kmeans.fit(data)
    
    # Get predictions
    predicted_labels = kmeans.labels
    
    # Print results
    print("K-Means clustering completed!")
    print(f"Number of clusters: {kmeans.n_clusters}")
    print("Cluster centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"  Cluster {i}: {centroid}")
    
    # Count points in each cluster
    cluster_counts = [0] * kmeans.n_clusters
    for label in predicted_labels:
        cluster_counts[label] += 1
    
    print("\nPoints per cluster:")
    for i, count in enumerate(cluster_counts):
        print(f"  Cluster {i}: {count} points")
    