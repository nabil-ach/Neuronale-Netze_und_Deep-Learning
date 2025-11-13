import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score

# Iris-Datensatz laden
iris = load_iris()
X = iris.data
y = iris.target

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell 1: Komplexes Modell (Overfitting)
tree_complex = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_complex.fit(X_train, y_train)

# Modell 2: Normales Modell
tree_normal = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_normal.fit(X_train, y_train)

# Perfomance Bewertung
print("Complex Model Training Accuracy:", tree_complex.score(X_train, y_train))
print("Complex Model Test Accuracy:", tree_complex.score(X_test, y_test))   
print("Normal Model Training Accuracy:", tree_normal.score(X_train, y_train))
print("Normal Model Test Accuracy:", tree_normal.score(X_test, y_test))

# Lernkurven erstellen
def plot_learning_curve(model, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
    plt.legend(loc="best")
    plt.show()

plot_learning_curve(tree_complex, "Learning Curve (Complex Model)")
plot_learning_curve(tree_normal, "Learning Curve (Normal Model)")