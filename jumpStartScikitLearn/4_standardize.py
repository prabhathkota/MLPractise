# Standardize the data attributes for the Iris dataset.
# Standardization (or Z-score normalization) refers to shifting the distribution of each attribute to have a
# mean of 0 and a standard deviation of 1.
# It is useful to standardize attributes for a model that relies on
# the distribution of attributes such as Gaussian processes.
# Standardization assumes that your data has a Gaussian (bell curve) distribution.
# Standardization (or Z-score normalization) is the process of rescaling the features so that they’ll have
# the properties of a Gaussian distribution with mean(μ=0) and standard deviation(σ=1)

from sklearn.datasets import load_iris
from sklearn import preprocessing

# load the Iris dataset
iris = load_iris()
print(iris.data.shape)

# separate the data and target attributes
X = iris.data
y = iris.target

# standardize the data attributes
standardized_X = preprocessing.scale(X)
print(standardized_X)
