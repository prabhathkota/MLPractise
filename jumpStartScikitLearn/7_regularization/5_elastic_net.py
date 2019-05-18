# ElasticNet Regression
# The ElasticNet is a regularized form of regression that penalizes the model with both the L1 norm and the L2 norm.
import numpy as np
from sklearn import datasets
from sklearn.linear_model import ElasticNet

# load the diabetes datasets
dataset = datasets.load_diabetes()

# fit a model to the data
model = ElasticNet(alpha=0.1)
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))

