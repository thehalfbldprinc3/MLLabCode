# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset (Age, Glucose Level, Diabetes)
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'GlucoseLevel': [80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170],
    'Diabetes': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 0 = No Diabetes, 1 = Diabetes
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Independent variables (features)
X = df[['Age', 'GlucoseLevel']]
# Dependent variable (target)
y = df['Diabetes']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict for new input
age = float(input("\nEnter age: "))
glucose = float(input("Enter glucose level: "))
prediction = model.predict([[age, glucose]])
result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
print(f"\nPrediction for Age = {age} and Glucose Level = {glucose}: {result}")

# Visualizing the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['GlucoseLevel'], c=df['Diabetes'], cmap='bwr', edgecolors='k')
plt.title('Diabetes Prediction based on Age and Glucose Level')
plt.xlabel('Age')
plt.ylabel('Glucose Level')

# Plot decision boundary
x_min, x_max = X['Age'].min() - 1, X['Age'].max() + 1
y_min, y_max = X['GlucoseLevel'].min() - 1, X['GlucoseLevel'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.colorbar()

plt.show()