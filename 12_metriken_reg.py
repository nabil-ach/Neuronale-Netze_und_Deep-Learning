from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Daten laden
data = fetch_california_housing()
X, y = data.data, data.target

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell erstellen
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Metriken berechnen
mae = mean_absolute_error(y_test, y_pred)                 # MAE
r2 = r2_score(y_test, y_pred)                             # R²-Score

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R²-Score: {r2}")