from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import matplotlib.pyplot as plt

# Hyperparameter
EPOCHS = 30
hidden_neurons = 50
# Bezeichner der Optimierungsverfahren
optimizers = ['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam', 'Nadam']

# Daten laden und vorverarbeiten
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)/255.0
x_test = x_test.reshape(10000, 784)/255.0

y_train_1hot = to_categorical(y_train, 10)
y_test_1hot = to_categorical(y_test, 10)

# Ergebnisse speichern
results = []

# Verschiedene Optimierer testen
def processFNN(optimizer):
    print(f'\n+++++++++++ {optimizer} ++++++++++')
    fnn_data = {} # Daten speichern
    fnn_data['title'] = optimizer
    
    model = Sequential()
    model.add(Dense(hidden_neurons, 
                    input_shape=(784,), 
                    activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['acc'])
    
    # Training
    start_time = time.time()
    history = model.fit(x_train, y_train_1hot, 
                        epochs=EPOCHS,
                        verbose=0,
                        validation_split=0.1)
    fnn_data['history'] = history
    
    # Performance auf Testdaten
    test = model.evaluate(x_test, y_test_1hot)
    fnn_data['test'] = test
    duration = time.time() - start_time 
    print(f'Dauer {duration:.2f} Sek | {EPOCHS} Epochen | Acc Val {test[1]:.3f}')
    return fnn_data

for op in optimizers:
    results.append(processFNN(op))

# Ergebnisse zusammenfassen
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

for res in results:
    ep = range(1, len(res['history'].history['acc'])+1 )
    ax[0].plot(ep, res['history'].history['acc'], label=res['title'])
    ax[0].set_ylabel('Acc')
    ax[0].set_xlabel('Epochen')
    ax[0].legend()
    ax[0].set_title('Accuracy Training')
    ax[0].grid()
    ax[1].plot(ep, res['history'].history['val_acc'], label=res['title'])
    ax[1].set_ylabel('Val Acc')
    ax[1].set_xlabel('Epochen')
    ax[1].grid()
    ax[1].set_title('Accuracy Validation')
    ax[1].legend()

plt.show()