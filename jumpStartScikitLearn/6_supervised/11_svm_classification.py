# SVM Classification
# Support Vector Machines (SVM) are a method that uses points in a transformed problem space that best separate
# classes into two groups.
# Classification for multiple classes is supported by a one­vs­all method. SVM also supports regression
# by modeling the function with a minimum amount of allowable error.

from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC

# load the iris datasets
dataset = datasets.load_iris()

# fit a SVM model to the data
model = SVC()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

