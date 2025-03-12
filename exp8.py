# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
# ------------------------- LOADING DATA -------------------------
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Display sample data
print("\nSample Data:\n", df.head())
# ------------------------- DATA PREPROCESSING -------------------------
# Scale the data for better clustering results
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
# ------------------------- APPLYING DBSCAN -------------------------
# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)
# Add cluster labels to the dataset
df['cluster'] = clusters
# ------------------------- VISUALIZATION (PCA) -------------------------
# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]
# Plot clusters using PCA components
plt.figure(figsize=(8, 5))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df, palette='viridis', s=100)
plt.title('DBSCAN Clustering on Iris Dataset (PCA Reduced)')
plt.show()
# ------------------------- EVALUATION -------------------------
# Number of clusters and noise points
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"\nNumber of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
# Display cluster assignments
print("\nCluster Assignments:")
print(df[['PC1', 'PC2', 'cluster']].head(10))