from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Create non-circular (moon-shaped) data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Fit DBSCAN
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

# Number of clusters in labels, ignoring noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters)
print('Estimated number of noise points: %d' % n_noise)

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title("DBSCAN Clustering")
plt.colorbar(label='Cluster label')
plt.savefig('dbscan_result.png')
print('Plot saved as dbscan_result.png')