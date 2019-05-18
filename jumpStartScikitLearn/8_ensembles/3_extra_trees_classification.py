# Extra Trees Classification
# The Extra Trees ensemble method creates a large number of random decision trees, each on a different
# subset of the dataset.
# Unlike Random Forest, the split points in the decision trees are selected randomly (i.e. random trees).
# Like Random Forest, this general ensemble method is called bagging.
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

# load the iris datasets
dataset = datasets.load_iris()

# fit a Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

