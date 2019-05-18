# Perceptron
# The Perceptron is a primitive type of neural network that learns weights for input attributes and transfers
# the weighted inputs into a network output or prediction.

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron

# load the diabetes datasets
dataset = datasets.load_diabetes()

# fit a Perceptron model to the data
model = Perceptron()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
print(expected)
predicted = model.predict(dataset.data)
print(predicted)

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))
