# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# ------------------------- LOADING DATA -------------------------
# Load the IRIS dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)
# Display first few rows of the dataset
print("\nSample Data:\n", df.head())
# ------------------------- DATA EXPLORATION -------------------------
# Basic information about the data
print("\nDataset Info:")
print(df.info())
# Summary statistics of numeric features
print("\nStatistical Summary:")
print(df.describe())
# Count of each class
print("\nClass Distribution:\n", df['species'].value_counts())
# ------------------------- VISUALIZATION -------------------------
# Pair plot to visualize relationships between features
sns.pairplot(df, hue='species')
plt.show()
# ------------------------- DATA PREPROCESSING -------------------------
# Splitting data into features (X) and target (y)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
# Splitting into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ------------------------- MODEL TRAINING -------------------------
# Creating and training the Decision Tree classifier using ID3 algorithm (criterion='entropy')
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)
# ------------------------- PREDICTION -------------------------
# Making predictions
y_pred = model.predict(X_test)
# ------------------------- MODEL EVALUATION -------------------------
# Evaluating the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ------------------------- PLOTTING -------------------------
# Plotting the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title('Decision Tree using ID3 Algorithm (IRIS Dataset)')
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