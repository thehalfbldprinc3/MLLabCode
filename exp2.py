# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (Hours Studied vs Marks Scored)
data = {
    'HoursStudied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'MarksScored': [35, 40, 45, 50, 55, 65, 70, 75, 80, 90]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Splitting the data into training and testing sets
X = df[['HoursStudied']]  # Independent variable
y = df['MarksScored']     # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Visualizing the training set results
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='blue', label='Best Fit Line')
plt.title('Marks vs Hours Studied (Training Set)')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.show()

# Visualizing the test set results
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_train, model.predict(X_train), color='blue', label='Best Fit Line')
plt.title('Marks vs Hours Studied (Test Set)')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.show()

# Model Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Coefficients
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Prediction example
hours = float(input("\nEnter hours studied: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for {hours} hours of study: {predicted_marks[0]:.2f}")