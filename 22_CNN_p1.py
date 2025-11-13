from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np  # <-- neu: für argmax / expand_dims

# Daten laden
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Modell
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
monitor = EarlyStopping(monitor='val_acc',
                        mode='max',
                        restore_best_weights=True,
                        patience=5)

# Training
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, callbacks=[monitor])

# Testen
# --- Änderung: predict benötigt eine Batch-Dimension; verwende [0:1] oder np.expand_dims ---
y_pred = model.predict(X_test[0:1])
predicted_class = np.argmax(y_pred, axis=1)[0]
print("Vorhersage (Wahrscheinlichkeiten) für das erste Testbild:", y_pred)
print("Vorhergesagte Klasse für das erste Testbild:", int(predicted_class))
