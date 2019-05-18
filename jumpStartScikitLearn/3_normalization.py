# Normalize the data attributes for the Iris dataset.
# Normalization refers to rescaling real valued numeric attributes into the range 0 and 1
# Normalization often also simply called Min-Max scaling
# It works better for cases in which the standardization might not work so well.
# If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.

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
