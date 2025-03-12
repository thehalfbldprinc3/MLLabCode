import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load example dataset
tips = sns.load_dataset('tips')

# 1. Line Plot
sns.set_theme(style="whitegrid")
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
sns.lineplot(x=x, y=y, color="blue", label="sin(x)")
plt.title("Line Plot Example")
plt.legend()
plt.show()

# 2. Scatter Plot
plt.figure(figsize=(8, 4))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', size='size', palette='cool', sizes=(20, 200))
plt.title("Scatter Plot Example")
plt.show()

# 3. Bar Plot
plt.figure(figsize=(8, 4))
sns.barplot(x='day', y='total_bill', data=tips, palette='viridis', ci=None)
plt.title("Bar Plot Example")
plt.show()

# 4. Histogram (using `displot`)
sns.displot(tips['total_bill'], bins=20, kde=True, color="purple", height=4, aspect=1.5)
plt.title("Histogram Example")
plt.show()

# 5. Box Plot
plt.figure(figsize=(8, 4))
sns.boxplot(x='day', y='total_bill', data=tips, palette='coolwarm')
plt.title("Box Plot Example")
plt.show()

# 6. Violin Plot
plt.figure(figsize=(8, 4))
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True, palette='pastel')
plt.title("Violin Plot Example")
plt.show()

# 7. Heatmap
corr = tips.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Example")
plt.show()

# 8. Pair Plot
sns.pairplot(tips, hue='sex', palette='husl')
plt.title("Pair Plot Example")
plt.show()

# 9. Count Plot
plt.figure(figsize=(8, 4))
sns.countplot(x='day', data=tips, hue='sex', palette='cool')
plt.title("Count Plot Example")
plt.show()

# 10. KDE Plot
plt.figure(figsize=(8, 4))
sns.kdeplot(tips['total_bill'], shade=True, color='red', bw_adjust=0.5)
plt.title("KDE Plot Example")
plt.show()

# 11. Lmplot (Regression Plot)
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', aspect=1.5, palette='coolwarm')
plt.title("Linear Regression Example")
plt.show()

# 12. Swarm Plot
plt.figure(figsize=(8, 4))
sns.swarmplot(x='day', y='total_bill', data=tips, hue='sex', palette='cool')
plt.title("Swarm Plot Example")
plt.show()