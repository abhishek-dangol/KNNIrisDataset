import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

### in the form of a dictionary###
iris = load_iris()

# transpose of data to separate into list of lists with each variable###
features = iris.data.T
###print(iris.data.T)###
sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

# extract the appropriate labels
sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

plt.scatter(sepal_length, sepal_width, c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)

plt.show()
