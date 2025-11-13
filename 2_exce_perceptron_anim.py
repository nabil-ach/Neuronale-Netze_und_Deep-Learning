import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Zufällige Daten erzeugen
np.random.seed(0)
X = np.random.rand(200, 2) * 10
y = (X[:, 1] > 0.5 * X[:, 0] + 1).astype(int)

# Anfangswerte
weights = np.random.rand(2)
bias = np.random.rand(1)
learning_rate = 0.01
epochs = 30  # Weniger Epochen, damit man es sieht

# Plot vorbereiten
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
line, = ax.plot([], [], 'g--', linewidth=2)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Perzeptron Lernprozess")

# Funktion zum Zeichnen der Linie
def update_line():
    x_vals = np.array([0, 10])
    y_vals = -(weights[0] / weights[1]) * x_vals - (bias / weights[1])
    line.set_data(x_vals, y_vals)

# Animationsfunktion (wird pro Frame aufgerufen)
def animate(epoch):
    global weights, bias
    for xi, target in zip(X, y):
        output = np.dot(xi, weights) + bias
        prediction = 1 if output >= 0 else 0
        error = target - prediction
        weights += learning_rate * error * xi
        bias += learning_rate * error
    update_line()
    ax.set_title(f"Epoch: {epoch+1}")
    return line,

# Animation erstellen
ani = FuncAnimation(fig, animate, frames=epochs, interval=500, repeat=False)
plt.show()
