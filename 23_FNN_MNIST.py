import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data and linearize
X_train_nz = X_train.astype('float32') / 255.0
X_train_nz = X_train_nz.reshape(-1, 28*28)

X_test_nz = X_test.astype('float32') / 255.0
X_test_nz = X_test_nz.reshape(-1, 28*28)

# One hot encode labels
y_train_oh = to_categorical(y_train, 10)
y_test_oh = to_categorical(y_test, 10)

# Model
model = Sequential([
    Input(shape=(784,)),
    Dense(60, activation='sigmoid'),
    Dense(10, activation='softmax')
])
model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['acc'])

# Fit model
start_time = time.time()
history = model.fit(X_train_nz, y_train_oh, epochs=20, batch_size=6000, verbose=1, validation_data=(X_test_nz, y_test_oh))
duration = time.time() - start_time
print(f"Training duration: {duration:.2f} seconds")

# Plot training & validation accuracy values
def set_subplot(ax, y_label, traindata, testdata, ylim):
    e_range = range(1, len(traindata) + 1)
    ax.plot(e_range, traindata, 'b', label='Training')
    ax.plot(e_range, testdata, 'g', label='Test')
    ax.set_xlabel('Epochen')
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid()
    ax.set_ylim(ylim)
    ax.set_title(y_label)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

set_subplot(ax[0], 'Loss', history.history['loss'], 
            history.history['val_loss'], [0, 3])
set_subplot(ax[1], 'Accuracy', history.history['acc'], 
            history.history['val_acc'], [0, 1])

plt.show()