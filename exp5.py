# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# ------------------------- LOADING DATA -------------------------
# Load the IRIS dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
# Display sample data
print("\nSample Data:\n", df.head())
# ------------------------- DATA EXPLORATION -------------------------
# Basic info about the dataset
print("\nDataset Info:")
print(df.info())
# Count of each class
print("\nClass Distribution:\n", df['species'].value_counts())
# Pair plot for visualization
sns.pairplot(df, hue='species')
plt.show()
# ------------------------- DATA PREPROCESSING -------------------------
# Split data into features (X) and target (y)
X = df.iloc[:, :-1]
y = df['species']
# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ------------------------- MODEL TRAINING -------------------------
# Creating and training the KNN classifier (with k=5)
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
# ------------------------- PREDICTION -------------------------
# Making predictions
y_pred = model.predict(X_test)
# ------------------------- MODEL EVALUATION -------------------------
# Accuracy score
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ------------------------- VISUALIZATION -------------------------
# Visualizing the decision boundaries (for sepal length and width)
from matplotlib.colors import ListedColormap
X_plot = X.iloc[:, [0, 1]].values
y_plot = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values
# Fitting the model for plotting
knn_plot = KNeighborsClassifier(n_neighbors=k)
knn_plot.fit(X_plot, y_plot)
# Generating mesh grid
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Predict on the mesh grid
Z = knn_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot decision boundary
plt.figure(figsize=(10, 6))
cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=cmap_points, edgecolor='k')
plt.title(f'KNN Decision Boundaries (k = {k})')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
# ------------------------- TESTING NEW DATA -------------------------
# Predict for new input
sepal_length = float(input("\nEnter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))
# Prediction
new_prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
print(f"\nPredicted Species: {new_prediction[0]}")