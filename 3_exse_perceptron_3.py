from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

# Prepare the data
X = iris.data.T
X = X[[0, 2]] 
X = X.T
y = iris.target

# only keep classes 0 and 1
X = X[y != 2]
y = y[y != 2]

# Visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[2])
# plt.title("Iris Dataset (Classes 0 and 1)")
# plt.show()

# Data as np.array
X = np.array(X)
y = np.array(y)

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Class Perceptron
class Perceptron:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
    # Train the perceptron
    def fit(self, x, y, epochs = 5):
        ran = np.random.RandomState(42)
        n_samples = x.shape[0]
        print(f"Train on {n_samples} samples.")
        # Gewichte: 1+ dim(x) für Bias
        self.w = ran.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        # Speicher für Verluste
        self.losses = []
        # Training
        for i in range(epochs):
            z = self.net_input(x)
            y_pred = self.activation(z)
            errors = (y - y_pred)
            # Update weights
            self.w[1:] += self.alpha * x.T.dot(errors)
            self.w[0] += self.alpha * errors.sum()
            # Calculate loss
            loss = (errors**2).sum() / 2.0
            self.losses.append(loss)
            print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")
        return self
    # Aktivierungsfunktion
    def activation(self, z):
        return z
    # Net input
    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]
    # Predict
    def predict(self, x):
        z = self.net_input(x)
        return np.where(z >= 0.0, 1, 0)
    
# Create and train the perceptron
EPOCHS = 20
model1 = Perceptron(alpha=0.01)
model1.fit(X_std, y, epochs=EPOCHS)

model2 = Perceptron(alpha=0.0001)
model2.fit(X_std, y, epochs=EPOCHS)

# Plot the loss over epochs
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax[0].plot(range(1, len(model1.losses)+1), np.log10(model1.losses), marker='o')
ax[0].set_xlabel('Epochen')
ax[0].set_ylabel('log(SSE)')
ax[0].set_title('Perceptron, alpha=0.01')

ax[1].plot(range(1, len(model2.losses)+1), np.log10(model2.losses), marker='o')
ax[1].set_xlabel('Epochen')
ax[1].set_ylabel('log(SSE)')
ax[1].set_title('Perceptron, alpha=0.0001')

plt.show()