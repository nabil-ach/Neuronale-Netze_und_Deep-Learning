import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Daten laden
iris = load_iris()
X, y = iris.data, iris.target

# Daten vorbereiten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Modell 1: Zu komplex (Overfitting)
complex_model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
complex_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modell 2: Mit Dropout und EarlyStopping
simple_model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
simple_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping-Callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history_complex = complex_model.fit(
    X_train, y_train, epochs=100, validation_split=0.2, verbose=0
)
history_simple = simple_model.fit(
    X_train, y_train, epochs=100, validation_split=0.2,
    callbacks=[early_stop], verbose=0
)

# Performance vergleichen
print("Komplexes Modell (Test):", complex_model.evaluate(X_test, y_test, verbose=0)[1])
print("Einfaches Modell (Test):", simple_model.evaluate(X_test, y_test, verbose=0)[1])

# # Trainingsverlauf plotten
# def plot_history(history, title):
#     plt.figure()
#     plt.title(title)
#     plt.plot(history.history['accuracy'], label='Train')
#     plt.plot(history.history['val_accuracy'], label='Validation')
#     plt.xlabel('Epochen')
#     plt.ylabel('Genauigkeit')
#     plt.legend()
#     plt.show()

# plot_history(history_complex, "Komplexes Modell (Overfitting)")
# plot_history(history_simple, "Einfaches Modell (mit Dropout/EarlyStopping)")
