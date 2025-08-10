import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate sample data
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
               linestyles=['dotted', 'solid', 'dashed'])

# Add labels and title
plt.title('GMM Soft Clustering Visualization', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(fontsize=10)

# Add text explaining the visualization
plt.figtext(0.5, 0.01, 
           'Soft clustering visualization with GMM.\n'
           'Contour lines show probability boundaries (30% dotted, 50% solid, 70% dashed) for each component.', 
           ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save the visualization
plt.savefig('gmm_simple_soft_clustering.png', dpi=200)
print("Simplified soft clustering visualization saved as 'gmm_simple_soft_clustering.png'")

