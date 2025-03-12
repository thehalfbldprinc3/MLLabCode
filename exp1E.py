from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load example dataset
data = load_iris()
X = data.data
y = data.target

# 1. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# 2. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("\nLinear Regression Coefficients:", lr.coef_)
print("Linear Regression Intercept:", lr.intercept_)

# 3. Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

# 4. Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_dt))

# 5. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))

# 6. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.show()

# 7. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# 8. Feature Importances (Random Forest)
importances = rf.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=data.feature_names, y=importances, palette="viridis")
plt.title("Feature Importances (Random Forest)")
plt.show()

# 9. Cross-Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

# 10. PCA (Principal Component Analysis)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm')
plt.title("PCA Example")
plt.show()

# 11. KMeans Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

plt.figure(figsize=(8, 4))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette="coolwarm", s=70)
plt.title("KMeans Clustering Example")
plt.show()

# 12. Decision Boundary Plot (Logistic Regression)
import numpy as np

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 4))
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, edgecolor='k', palette="coolwarm")
plt.title("Decision Boundary (Logistic Regression)")
plt.show()