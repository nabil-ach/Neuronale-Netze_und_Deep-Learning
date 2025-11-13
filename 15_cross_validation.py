from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Daten laden
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

# 1. Daten laden (mit Scaling!)
X_scaled = scaler.fit_transform(X)  # Scaling auf alle Daten anwenden

# 2. Modell definieren
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 3. 5-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')

# 4. Ergebnisse ausgeben
print("Accuracy pro Fold:", cv_scores)
print("Durchschnittliche Accuracy:", np.mean(cv_scores))
print("Standardabweichung:", np.std(cv_scores))
