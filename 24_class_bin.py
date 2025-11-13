import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# dataset loading
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Linearisation and normalisation
X_train = X_train.reshape((X_train.shape[0], 784)) / 255
X_test = X_test.reshape((X_test.shape[0], 784)) / 255

# Binary classification for digit '5'
y_train_5 = (y_train == 5).astype("float32")
y_test_5 = (y_test == 5).astype("float32")

# model definition
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(784,))
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])

# model training
history1 = model.fit(X_train, y_train_5, epochs=10, validation_data=(X_test, y_test_5))

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype("float32")

# Confusion matrix
cm = confusion_matrix(y_test_5, y_pred_labels)
print("Confusion Matrix:")
print(cm)

# Precision, Recall, Accuracy
precision = precision_score(y_test_5, y_pred_labels)
recall = recall_score(y_test_5, y_pred_labels)
accuracy = accuracy_score(y_test_5, y_pred_labels)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
