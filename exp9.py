# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
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
# ------------------------- APPLYING K-MEDOIDS -------------------------
# Apply K-Medoids clustering with 3 clusters (since we have 3 iris species)
kmedoids = KMedoids(n_clusters=3, random_state=42)
clusters = kmedoids.fit_predict(scaled_data)
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
plt.scatter(pca_result[kmedoids.medoid_indices_, 0], 
            pca_result[kmedoids.medoid_indices_, 1], 
            color='red', label='Medoids', s=150, marker='X')
plt.title('K-Medoids Clustering on Iris Dataset (PCA Reduced)')
plt.legend()
plt.show()
# ------------------------- EVALUATION -------------------------
# Number of clusters and size of each cluster
n_clusters = len(set(clusters))
cluster_sizes = pd.Series(clusters).value_counts()
print(f"\nNumber of clusters: {n_clusters}")
print("\nCluster Sizes:")
print(cluster_sizes)
# Display cluster assignments
print("\nCluster Assignments:")
print(df[['PC1', 'PC2', 'cluster']].head(10))