# SVM Regression
# Support Vector Machines (SVM) are a method that uses points in a transformed problem space that best separate
# classes into two groups.
# Classification for multiple classes is supported by a one­vs­all method. SVM also supports regression
# by modeling the function with a minimum amount of allowable error.

import numpy as np
from sklearn import datasets
from sklearn.svm import SVR

# load the datasets
dataset = datasets.load_diabetes()

# fit a SVM model to the data
model = SVR()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))

