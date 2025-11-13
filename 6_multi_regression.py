import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as lm

# Daten einlesen
df = pd.read_csv("lerning_code\data.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']

# Modell erstellen
regr = lm.LinearRegression()
regr.fit(X, y)

# print
print("Koeffizienten:", regr.coef_)  # Koeffizienten für Weight und Volume
print("Achsenabschnitt (Intercept):", regr.intercept_)  # Achsenabschnitt