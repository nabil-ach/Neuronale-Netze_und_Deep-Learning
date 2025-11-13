import numpy as np
import matplotlib.pyplot as plt

# Beispiel-Daten generieren # Linie: y = 0.5*x + 1
#np.random.seed(0)
X = np.random.rand(2000, 2) * 10
y = (X[:,1] > 0.5 * X[:,0] + 1).astype(int)  # Gleichung w1*x1 + w2*x2 + b = 0

# Perzeptron-Parameter
weights = np.random.rand(2)
bias = float(np.random.rand())
lerning_rate = 0.01
epochs = 100

print("Initiale Gleichung:")
print("y = {:.2f}*x + {:.2f}".format(-weights[0]/weights[1], -bias/weights[1]))

# Trainingsprozess
for _ in range(epochs):
    for xi, target in zip(X, y):
        out = np.dot(xi, weights) + bias # out = x0*w0 + x1*w1 + b || w0 = -0,5; w1 = 1; b = -1
        prediction = 1 if out >= 0 else 0
        error = target - prediction
        weights += lerning_rate * error * xi    # weight update
        bias += lerning_rate * error            # bias update

print("Trainierte Gleichung:")
print("y = {:.2f}*x + {:.2f}".format(-weights[0]/weights[1], -bias/weights[1]))

# Ergebnis visualisieren
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
x_vals = np.array([0, 10])
y_vals = -(weights[0]/weights[1])*x_vals - (bias/weights[1])
plt.plot(x_vals, y_vals, 'g--') 
plt.show()
