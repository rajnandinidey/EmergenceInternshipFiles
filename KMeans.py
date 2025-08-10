import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.75, random_state=33)

# Step 2: Fit KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=33)
kmeans.fit(X)

# Step 3: Get predictions and centroids
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# Step 4: Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("KMeans Clustering with scikit-learn", fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot to a file 
output_file = 'kmeans_result.png'
plt.savefig(output_file)
plt.close()
print(f"Plot saved to {output_file}")

# Print some information about the clustering
print(f"\nNumber of clusters: {len(centroids)}")
print(f"Cluster centers:\n{centroids}")

# Count the number of points in each cluster
unique, counts = np.unique(y_kmeans, return_counts=True)
for i, (cluster, count) in enumerate(zip(unique, counts)):
    print(f"Cluster {cluster}: {count} points")
