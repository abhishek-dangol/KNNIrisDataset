from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()
# random state is like a seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

# to build the model
knn.fit(X_train, y_train)

# test data with random values
X_new = np.array([[5, 2.9, 1.0, 0.2]])

prediction = knn.predict(X_new)

# 0 for iris setosa, 1 for iris versicolor and 2 for iris virginica
print(prediction)

# accuracy of the model
print(knn.score(X_test, y_test))
