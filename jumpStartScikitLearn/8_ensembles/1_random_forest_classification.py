# Random Forest Classification
# Ensemble methods take the predictions from two or more models and combine them into a single prediction.
# In this way, ensemble methods are models of models or meta­models.
# They are a powerful set of techniques for getting better results from existing models.
# Decision trees like CART and randomly created trees are a popular base model for creating ensembles.

# The Random Forest ensemble method creates a number of decision trees, each on a different subset of the dataset.
# Generally the approach to creating an ensemble used by random forest is called bootstrap aggregation
# (or bagging for short).
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# load iris the datasets
dataset = datasets.load_iris()

# fit a random forest model to the data
model = RandomForestClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))