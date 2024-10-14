import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv('feats.csv')  # Replace with your CSV file name

# Preprocess the data
X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find optimal number of clusters
inertias = []
max_clusters = 10

for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Print the inertia values
for k, inertia in enumerate(inertias, start=1):
    print(f"Clusters: {k}, Inertia: {inertia}")

# Based on the elbow curve, choose the optimal number of clusters
optimal_clusters = 4  # Replace this with the number you determine from the elbow curve

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42,n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

# Plot the t-SNE results with cluster colors
plt.figure(figsize=(10, 8),dpi=300)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of clusters')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()