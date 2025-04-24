# ------------------------- IMPORTING NECESSARY LIBRARIES -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------- LOADING DATA -------------------------
# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Display dataset shape and feature names
print("Dataset shape:", X.shape)
print("Target classes:", target_names)

# ------------------------- DATA PREPROCESSING -------------------------
# Scale the data for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ------------------------- TRAINING THE SVM MODEL -------------------------
# Create and train the Support Vector Classifier
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(x_train, y_train)

# Predict on test set
predictions = svm_model.predict(x_test)

# ------------------------- EVALUATION -------------------------
# Print classification results
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=target_names))

# ------------------------- VISUALIZATION USING PCA -------------------------
# Reduce features to 2 principal components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for plotting
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Plot PCA-reduced data
plt.figure(figsize=(8, 5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='Set1', s=100)
plt.title('Breast Cancer Dataset (PCA Reduced)')
plt.legend(title='Class', labels=target_names)
plt.show()