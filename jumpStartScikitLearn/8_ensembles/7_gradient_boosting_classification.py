# Gradient Boosting Classification
# Gradient Boosting is an ensemble method that uses boosted decision trees like AdaBoost,
# but is generalized to allow an arbitrary differentiable loss function (i.e. how model error is calculated).
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

# load the iris datasets
dataset = datasets.load_iris()

# fit a Gradient Boosting model to the data
model = GradientBoostingClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

