# LassoLars Regression
# The Least Angle Regression (LARS) can be used as an alternative method for calculating Least Absolute Shrinkage
# and Selection Operator (LASSO) fit.
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LassoLars

# load the iris datasets
dataset = datasets.load_diabetes()

# fit a LASSO using LARS model to the data
model = LassoLars(alpha=0.1)
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))

