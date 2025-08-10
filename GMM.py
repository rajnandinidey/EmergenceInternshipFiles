import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generating synthetic data
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.7, random_state=40)

# Fit GMM
gmm = GaussianMixture(n_components=5, random_state=40)
gmm.fit(X)

# Predict cluster probabilities and labels
probs = gmm.predict_proba(X)
labels = gmm.predict(X)

# Create a mesh grid for contour plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
XY = np.column_stack([xx.ravel(), yy.ravel()])

# Get probabilities for all points in the mesh grid
Z_all = gmm.predict_proba(XY)

# Create a figure for the visualization
plt.figure(figsize=(10, 8))

# Define colors for each component
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Plot data points with their assigned cluster color
plt.scatter(X[:, 0], X[:, 1], c=[colors[l] for l in labels],
          s=50, alpha=0.7, edgecolors='w')

# Plot cluster centers
for i in range(5):
   plt.scatter(gmm.means_[i, 0], gmm.means_[i, 1],
             color='black', marker='X', s=150,
             edgecolors=colors[i], linewidths=2,
             label=f'Cluster {i+1} Center')
  
   # Plot three probability contours (0.3, 0.5, 0.7) for each component
   z = Z_all[:, i].reshape(xx.shape)
   plt.contour(xx, yy, z, levels=[0.3, 0.5, 0.7],
              colors=[colors[i]], alpha=0.8, linewidths=[1, 2, 1],
              linestyles=['dotted', 'dashed', 'solid'])

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title("GMM Clustering")
plt.legend()

# Save the plot to a file
plt.savefig('gmm_clustering.png')


