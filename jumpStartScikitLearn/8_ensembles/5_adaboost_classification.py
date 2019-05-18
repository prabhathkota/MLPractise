# AdaBoost Classification
# Adaptive Boosting (AdaBoost) is an ensemble method that sums the predictions made by multiple decision trees.
# Additional models are added and trained on the data instances that were miss­classified by
# previous models in the ensemble.
# Generally the approach to creating an ensemble used by AdaBoost is called boosting.
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

# load the iris datasets
dataset = datasets.load_iris()

# fit an AdaBoost model to the data
model = AdaBoostClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


