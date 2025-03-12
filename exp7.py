# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# ------------------------- LOADING DATA -------------------------
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
# Display sample data
print("\nSample Data:\n", df.head())
# ------------------------- DATA PREPROCESSING -------------------------
# Split data into features (X) and target (y)
X = df.iloc[:, :-1]
y = df['species']
# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ------------------------- PCA -------------------------
# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Convert PCA results to a DataFrame for visualization
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = y
# Plot PCA result
plt.figure(figsize=(8, 5))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='viridis', s=100)
plt.title('PCA - Iris Dataset (2D)')
plt.show()
# ------------------------- LDA -------------------------
# Apply LDA to reduce to 2 components
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
# Convert LDA results to a DataFrame for visualization
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
lda_df['species'] = y
# Plot LDA result
plt.figure(figsize=(8, 5))
sns.scatterplot(x='LD1', y='LD2', hue='species', data=lda_df, palette='coolwarm', s=100)
plt.title('LDA - Iris Dataset (2D)')
plt.show()
# ------------------------- MODEL TRAINING (ON PCA) -------------------------
# Training the Naive Bayes model on PCA-transformed data
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_pca = GaussianNB()
model_pca.fit(X_train_pca, y_train)
# Predict using PCA data
y_pred_pca = model_pca.predict(X_test_pca)
# Evaluate PCA model
print("\nPCA Model Accuracy:", accuracy_score(y_test, y_pred_pca))
print("\nConfusion Matrix (PCA):\n", confusion_matrix(y_test, y_pred_pca))
print("\nClassification Report (PCA):\n", classification_report(y_test, y_pred_pca))
# ------------------------- MODEL TRAINING (ON LDA) -------------------------
# Training the Naive Bayes model on LDA-transformed data
X_train_lda, X_test_lda, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
model_lda = GaussianNB()
model_lda.fit(X_train_lda, y_train)
# Predict using LDA data
y_pred_lda = model_lda.predict(X_test_lda)
# Evaluate LDA model
print("\nLDA Model Accuracy:", accuracy_score(y_test, y_pred_lda))
print("\nConfusion Matrix (LDA):\n", confusion_matrix(y_test, y_pred_lda))
print("\nClassification Report (LDA):\n", classification_report(y_test, y_pred_lda))