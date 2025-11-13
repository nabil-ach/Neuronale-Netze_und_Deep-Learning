import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("lerning_code\data - tree.csv")

# convert non numerical data to numerical
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'NO': 0, 'YES': 1}
df['Go'] = df['Go'].map(d)

# X und y Auswahl
X = df[['Age', 'Experience', 'Rank', 'Nationality']]
y = df['Go']

# Decision Tree Modell erstellen
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# plot tree
tree.plot_tree(dtree , feature_names=['Age', 'Experience', 'Rank', 'Nationality'])

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
plt.show()