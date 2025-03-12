import matplotlib.pyplot as plt
import numpy as np

# 1. Line Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="sin(x)", color="blue", linestyle="--")
plt.title("Line Plot Example")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.grid(True)
plt.show()

# 2. Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)
sizes = 300 * np.random.rand(50)
colors = np.random.rand(50)

plt.figure(figsize=(8, 4))
plt.scatter(x, y, s=sizes, c=colors, alpha=0.5, cmap="viridis")
plt.colorbar(label="Color Scale")
plt.title("Scatter Plot Example")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

# 3. Bar Plot
categories = ["A", "B", "C", "D", "E"]
values = [3, 7, 1, 8, 5]

plt.figure(figsize=(8, 4))
plt.bar(categories, values, color="skyblue", edgecolor="black")
plt.title("Bar Plot Example")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

# 4. Histogram
data = np.random.randn(1000)

plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, color="lightgreen", edgecolor="black", alpha=0.7)
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 5. Pie Chart
sizes = [15, 30, 45, 10]
labels = ["A", "B", "C", "D"]
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
plt.title("Pie Chart Example")
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# 6. Multiple Subplots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ax[0].plot(x, y1, color="blue", label="sin(x)")
ax[0].set_title("Sin Wave")
ax[0].legend()

ax[1].plot(x, y2, color="red", label="cos(x)")
ax[1].set_title("Cos Wave")
ax[1].legend()

plt.tight_layout()
plt.show()

# 7. Customizing Plots
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, color="green", linewidth=2, linestyle="-.")
plt.title("Customized Line Plot")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(-1, 1.5, 0.5))
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# 8. Adding Annotations
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="sin(x)", color="purple")
plt.annotate("Maximum", xy=(1.5 * np.pi, 1), xytext=(4, 1.2),
             arrowprops=dict(facecolor="red", shrink=0.05))
plt.title("Annotated Plot")
plt.legend()
plt.show()

# 9. Saving a Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="sin(x)", color="blue")
plt.title("Saving Plot Example")
plt.legend()
plt.savefig("saved_plot.png", dpi=300)  # Save as a PNG file
plt.show()

# 10. Polar Plot
theta = np.linspace(0, 2 * np.pi, 100)
r = np.abs(np.sin(2 * theta))

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, r, color="orange")
ax.set_title("Polar Plot Example")
plt.show()