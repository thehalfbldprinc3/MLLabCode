# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
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

# ------------------------- DATA EXPLORATION -------------------------

# Basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Count of each class
print("\nClass Distribution:\n", df['species'].value_counts())

# Pair plot for visualization
sns.pairplot(df, hue='species', height=2)
plt.show()

# ------------------------- DATA PREPROCESSING -------------------------

# Split data into features (X) and target (y)
X = df.iloc[:, :-1]
y = df['species']

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------- MODEL TRAINING -------------------------

# Creating and training the Naive Bayes classifier
model = GaussianNB()
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

# ------------------------- TESTING NEW DATA -------------------------

# Predict for new input
sepal_length = float(input("\nEnter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

# Pass input as a DataFrame to avoid warning
new_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                        columns=X.columns)

# Prediction
new_prediction = model.predict(new_data)
print(f"\nPredicted Species: {new_prediction[0]}")

# ------------------------- VISUALIZATION -------------------------

# ✅ Fix column names for Seaborn plotting
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df, s=100)
plt.title('Sepal Length vs Sepal Width')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=df, s=100)
plt.title('Petal Length vs Petal Width')
plt.show()