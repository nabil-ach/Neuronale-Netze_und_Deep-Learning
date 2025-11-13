import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Daten laden
df = sns.load_dataset("titanic")

# lokale Daten
X = df[['age', 'fare', 'pclass', 'sex']].dropna()
y = X.pop('pclass')

# Train-Test-Split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: One-Hot-Encoding für kategoriale Merkmale
num_features = ['age', 'fare']
cat_features = ['sex']

proprocessing = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# Modellpipeline
pipeline = Pipeline(steps=[
    ('preprocessor', proprocessing),
    ('classifier', LogisticRegression())
])

# Modelltraining
pipeline.fit(X_train, y_train)

# Vorhersagen
y_pred = pipeline.predict(X_test)

# Genauigkeit bewerten
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')