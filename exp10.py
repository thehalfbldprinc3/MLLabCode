# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# ------------------------- LOADING DATA -------------------------
# Load the Handwritten Digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Display sample data
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.axis('off')
plt.suptitle('Sample Handwritten Digits')
plt.show()
# ------------------------- DATA PREPROCESSING -------------------------
# Scale the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ------------------------- APPLYING K-MEANS -------------------------
# Apply K-Means clustering with 10 clusters (since we have digits 0-9)
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
# ------------------------- VISUALIZATION (PCA) -------------------------
# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Create a DataFrame for plotting
import pandas as pd
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['cluster'] = clusters
# Plot clusters using PCA components
plt.figure(figsize=(8, 5))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df, palette='tab10', s=100)
plt.title('K-Means Clustering on Handwritten Digits (PCA Reduced)')
plt.legend(title='Cluster')
plt.show()
# ------------------------- EVALUATION -------------------------
# Count the number of samples in each cluster
cluster_sizes = pd.Series(clusters).value_counts()
print("\nNumber of samples in each cluster:")
print(cluster_sizes)
# Display cluster assignments for the first 10 samples
print("\nCluster Assignments:")
print(df[['PC1', 'PC2', 'cluster']].head(10))
# ------------------------- DISPLAY CLUSTER CENTERS -------------------------
# Display cluster centers as images
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
plt.figure(figsize=(8, 4))
for i, center in enumerate(centers):
    plt.subplot(2, 5, i + 1)
    plt.imshow(center, cmap='gray')
    plt.axis('off')
plt.suptitle('Cluster Centers as Handwritten Digits')
plt.show()