import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

# Split data into training and testing sets
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

# polynomial fit
mypol = np.poly1d(np.polyfit(train_x, train_y, 4))

# r2 score
print("R2 score train data:", r2_score(train_y, mypol(train_x)))
print("R2 score test data:", r2_score(test_y, mypol(test_x)))

# Plot training and testing data
plt.scatter(train_x, train_y, color='b', label='Training Data')
plt.scatter(test_x, test_y, color='r', label='Testing Data')
plt.plot(np.sort(x), mypol(np.sort(x)), color='black', label='Polynomial Fit')
plt.legend()
plt.show()