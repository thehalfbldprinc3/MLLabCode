from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
print("LR Coefs:", lr.coef_)

# Logistic Regression
log_reg = LogisticRegression(max_iter=200).fit(X_train, y_train)
print("LogReg Acc:", accuracy_score(y_test, log_reg.predict(X_test)))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
print("DT Acc:", accuracy_score(y_test, dt.predict(X_test)))

# Random Forest
rf = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("RF Acc:", accuracy_score(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, xticklabels=data.target_names, yticklabels=data.target_names); plt.title("Confusion Matrix"); plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Feature Importance
sns.barplot(x=data.feature_names, y=rf.feature_importances_); plt.title("Feature Importances"); plt.show()

# Cross-Validation
scores = cross_val_score(rf, X, y, cv=5)
print("CV Score Mean:", scores.mean())

# PCA
X_pca = PCA(n_components=2).fit_transform(X)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y); plt.title("PCA"); plt.show()

# KMeans Clustering
clusters = KMeans(n_clusters=3).fit_predict(X)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters); plt.title("KMeans Clustering"); plt.show()

# Decision Boundary (LogReg)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y); plt.title("Decision Boundary"); plt.show()