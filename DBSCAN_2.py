import math
import random


class DBSCAN:
    # DBSCAN clustering algorithm implementation from scratch.
    
    # Parameters:
    #    eps (float): The maximum distance between two samples for one to be considered
    #                as in the neighborhood of the other.
    #    min_samples (int): The number of samples in a neighborhood for a point
    #                       to be considered as a core point.
    #    labels (list): Cluster labels for each point in the dataset.
    #                  -1 represents noise points.
    
    def __init__(self, eps=0.5, min_samples=5):
        # Initialize the DBSCAN object with parameters.
        
        # Parameters:
        #    eps (float): The maximum distance between two samples for one to be considered
        #                as in the neighborhood of the other.
        #    min_samples (int): The number of samples in a neighborhood for a point
        #                       to be considered as a core point.
        
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def _euclidean_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points.
        
        # Parameters:
        #    point1 (list): First point coordinates
        #    point2 (list): Second point coordinates
            
        # Returns:
        #    float: Euclidean distance between the points
        
        # Sum of squared differences
        squared_diff_sum = 0
        for i in range(len(point1)):
            squared_diff_sum += (point1[i] - point2[i]) ** 2
        
        # Return the square root of the sum
        return math.sqrt(squared_diff_sum)
    
    def _get_neighbors(self, data, point_idx):
        # Find all points within eps distance of the point at point_idx.
        
        # Parameters:
        #    data (list): List of data points
        #    point_idx (int): Index of the point to find neighbors for
            
        # Returns:
        #    list: Indices of neighboring points
        
        neighbors = []
        for i in range(len(data)):
            if i != point_idx:  # Skip the point itself
                distance = self._euclidean_distance(data[point_idx], data[i])
                if distance <= self.eps:
                    neighbors.append(i)
        
        return neighbors
    
    def _expand_cluster(self, data, labels, point_idx, neighbors, cluster_id):
        # Expand the cluster from a core point.
        
        # Parameters:
        #    data (list): List of data points
        #    labels (list): Current cluster labels for each point
        #    point_idx (int): Index of the core point to expand from
        #    neighbors (list): Indices of points in the neighborhood of point_idx
        #    cluster_id (int): Current cluster ID
            
        # Returns:
        #    list: Updated cluster labels
        
        # Assign the cluster ID to the current point
        labels[point_idx] = cluster_id
        
        # Process each neighbor
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If the point was previously marked as noise, add it to the cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If the point is not yet assigned to a cluster
            elif labels[neighbor_idx] == 0:
                # Mark it as part of the current cluster
                labels[neighbor_idx] = cluster_id
                
                # Find the neighbors of this point
                new_neighbors = self._get_neighbors(data, neighbor_idx)
                
                # If this is a core point, add its neighbors to be processed
                if len(new_neighbors) >= self.min_samples:
                    for new_neighbor in new_neighbors:
                        if new_neighbor not in neighbors:
                            neighbors.append(new_neighbor)
            
            i += 1
        
        return labels
    
    def fit(self, data):
        # Fit the DBSCAN model to the data.
        
        # Parameters:
        #    data (list): List of data points
            
        # Returns:
        #    self: The fitted model
        
        n_points = len(data)
        
        # Initialize all points as unvisited (0)
        # -1 will represent noise points, and positive integers will be cluster IDs
        self.labels = [0] * n_points
        
        # Start with cluster ID 1
        cluster_id = 1
        
        # Process each point
        for i in range(n_points):
            # Skip points that have already been processed
            if self.labels[i] != 0:
                continue
            
            # Find neighbors within eps distance
            neighbors = self._get_neighbors(data, i)
            
            # If the point doesn't have enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                # Expand the cluster from this core point
                self.labels = self._expand_cluster(data, self.labels, i, neighbors, cluster_id)
                
                # Move to the next cluster
                cluster_id += 1
        
        return self


# Example usage and visualization
if __name__ == "__main__":
    # Function to generate moon-shaped data
    def generate_moon_data(n_samples=300, noise=0.05):
        # Generate moon-shaped data.
        
        # Parameters:
        #    n_samples (int): Number of samples to generate
        #    noise (float): Standard deviation of Gaussian noise added to the data
            
        # Returns:
        #    list: Generated data points
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Generate the outer half circle
        outer_circ_x = []
        outer_circ_y = []
        for i in range(n_samples_out):
            angle = math.pi * i / n_samples_out
            x = math.cos(angle)
            y = math.sin(angle)
            # Add some noise
            x += random.gauss(0, noise)
            y += random.gauss(0, noise)
            outer_circ_x.append(x)
            outer_circ_y.append(y)
        
        # Generate the inner half circle
        inner_circ_x = []
        inner_circ_y = []
        for i in range(n_samples_in):
            angle = math.pi * i / n_samples_in
            x = 1 - math.cos(angle)
            y = 1 - math.sin(angle) - 0.5
            # Add some noise
            x += random.gauss(0, noise)
            y += random.gauss(0, noise)
            inner_circ_x.append(x)
            inner_circ_y.append(y)
        
        # Combine the data
        x = outer_circ_x + inner_circ_x
        y = outer_circ_y + inner_circ_y
        
        # Convert to list of points
        data = [[x[i], y[i]] for i in range(len(x))]
        
        return data
    
    # Generate moon-shaped data
    data = generate_moon_data(n_samples=300, noise=0.05)
    
    # Fit DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(data)
    
    # Get cluster labels
    labels = dbscan.labels
    
    # Count clusters and noise points
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = labels.count(-1)
    
    # Print results
    print("DBSCAN clustering completed!")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
    
    # Count points in each cluster
    cluster_counts = {}
    for label in labels:
        if label not in cluster_counts:
            cluster_counts[label] = 0
        cluster_counts[label] += 1
    
    print("\nPoints per cluster:")
    for label, count in cluster_counts.items():
        if label == -1:
            print(f"  Noise: {count} points")
        else:
            print(f"  Cluster {label}: {count} points")
    