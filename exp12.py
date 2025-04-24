# ------------------------- IMPORTING NECESSARY LIBRARIES -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# ------------------------- LOADING AND PREPROCESSING DATA -------------------------
# Load the wine dataset (also available on Kaggle)
wine_data = load_wine()
X = wine_data.data
feature_names = wine_data.feature_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------- ELBOW METHOD TO FIND OPTIMAL K -------------------------
wcss = []

# Test for cluster counts from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# ------------------------- PLOTTING THE ELBOW GRAPH -------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='green')
plt.title('Elbow Method for Optimal K (Wine Dataset)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()