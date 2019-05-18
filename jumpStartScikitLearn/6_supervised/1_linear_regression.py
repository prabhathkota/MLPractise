# Linear Regression
# Linear regression fits a linear model (such as a line in two dimensions) to the data.

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd

# load the diabetes datasets
dataset = datasets.load_diabetes()
print(type(dataset.data))
#print(dataset)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
print(df.corr())

# fit a linear regression model to the data
model = LinearRegression()
model.fit(dataset.data, dataset.target)
# print(model)

print('--------')
# make predictions
expected = dataset.target
print(expected)
predicted = model.predict(dataset.data)
#print(predicted)
print('--------')

# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))
