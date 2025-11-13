from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Daten laden
iris = load_iris()
X, y = iris.data, iris.target

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Modell trainieren
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen
y_pred = model.predict(X_test)

# Metriken berechnen
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score (makro):", f1_score(y_test, y_pred, average='macro'))
print("Precision (makro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (makro):", recall_score(y_test, y_pred, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
