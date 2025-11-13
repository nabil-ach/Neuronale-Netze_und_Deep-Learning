import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
classes = [0,0,0,0,0,1,1,1,1,1]

data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(data, classes)

new_x = 5.6
new_y = 5.6
new_point = [(new_x, new_y)]

prediction = knn.predict(new_point)
print("Predicted:", prediction)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()