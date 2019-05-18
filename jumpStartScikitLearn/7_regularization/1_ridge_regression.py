# Ridge Regression
# Regularization refers to methods that decrease over­fitting of a model,
# most commonly by introducing a penalty proportional to model complexity.
# It is most common to demonstrate regularization methods using regression algorithms as the base model.

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge

# load the diabetes datasets
dataset = datasets.load_diabetes()

# fit a ridge regression model to the data
model = Ridge(alpha=0.1)
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))

