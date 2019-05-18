# Gaussian Naive Bayes
# Naive Bayes uses Bayes Theorem to model the conditional relationship of each attribute to the class variable.

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# load the iris datasets
dataset = datasets.load_iris()

# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))