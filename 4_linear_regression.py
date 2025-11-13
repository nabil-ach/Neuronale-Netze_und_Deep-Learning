import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# >60% Relationschip
a1 = [5,7,8,7,2,17,2,9,4,11,12,9,6]
b1 = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# ~0% Relationschip
a2 = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
b2 = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

# 100% Relationschip
a3 = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
b3 = 3.0 * a3 + 1  # Verwendet NumPy-Array-Operationen

# x y Auswahl
x = a1
y = b1


slop, intercept, r, p, std_err = stats.linregress(x, y)
print("r:", r ) # Korrelationskoeffizient
print("p:", p ) # p-Wert
print("std_err:", std_err ) # Standardfehler

def myfunc(x):
    return slop * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.plot(10, myfunc(10), color='red', marker='o')  # Vorhersage für x=10 (Prediction for x=10)
plt.show()