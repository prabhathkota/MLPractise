# Normalize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the iris dataset
iris = load_iris()
print(iris.data.shape)
# separate the data from the target attributes
X = iris.data
print(X)
y = iris.target
print(y)
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
print(normalized_X)
