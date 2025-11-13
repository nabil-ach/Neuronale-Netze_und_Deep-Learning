import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Beispiel-Daten
X = np.random.rand(100, 4)
y = tf.keras.utils.to_categorical(np.random.randint(3, size=100), num_classes=3)

# Modell erstellen
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),   
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modell trainieren
model.fit(X, y, epochs=10, verbose=0)

# Modell als .h5 speichern
model.save('model.h5')

# Konvertierung in TensorFlow Lite Modell
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Speichern des TFLite Modells
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)