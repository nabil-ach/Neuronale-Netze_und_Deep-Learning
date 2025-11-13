import numpy as np
import matplotlib.pyplot as plt

# Zufällige Beispielpunkte erzeugen (ähnlich wie deine blauen Punkte)
X = np.random.rand(30, 1) * 10  # 30 Punkte zwischen 0 und 10
y = 2.5 * X + np.random.randn(30, 1) * 2  # Lineare Beziehung mit Rauschen

# Anfangswerte für Gewichte und Bias
weights = np.random.randn()
bias = np.random.randn()
learning_rate = 0.01
epochs = 10000
n = len(X)

# Gewichte und Bias initial ausgeben
print("---------------------------------")
print("Initiale Gewichte und Bias:")
print(weights, bias)

# Trainingsprozess
for _ in range(epochs):
    y_pred = weights * X + bias
    cost = (1/n) * sum((y - y_pred) ** 2)
    # Gradienten berechnen
    dw = (-2/n) * sum(X * (y - y_pred))
    db = (-2/n) * sum(y - y_pred)
    # Gewichte und Bias aktualisieren
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Endwerte für Gewichte und Bias ausgeben
print("Trainierte Gewichte und Bias:")
print(weights, bias)
print("---------------------------------")

# Ergebnis visualisieren
plt.scatter(X, y, color = "blue", label="Datenpunkte")
plt.plot(X, weights * X + bias, color = "red", label="Lineare Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()