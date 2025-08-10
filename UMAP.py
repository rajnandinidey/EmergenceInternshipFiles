import umap
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load sample data (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Plot the data
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.title("UMAP Visualization of Iris Dataset")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig('umap_result.png')
plt.show()

