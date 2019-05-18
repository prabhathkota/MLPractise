# k-Nearest Neighbor Classification
# The k­Nearest Neighbor (kNN) method makes predictions by locating similar cases to a given data instance
# (using a similarity function) and returning the average or majority of the most similar data instances.
# The kNN algorithm can be used for classification or regression.

from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# load iris the datasets
dataset = datasets.load_iris()

# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

