# Lasso Regression
# The Least Absolute Shrinkage and Selection Operator (LASSO) is a regularization method that penalizes a
# least squares regression model on the absolute magnitude of the coefficients (called the L1 norm).

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Lasso

# load the diabetes datasets
dataset = datasets.load_diabetes()

# fit a LASSO model to the data
model = Lasso(alpha=0.1)
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))

