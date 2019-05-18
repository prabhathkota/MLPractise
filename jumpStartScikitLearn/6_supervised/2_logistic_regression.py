# Logistic Regression
#Logistic regression fits a logistic model to data and makes predictions about the
# probability of an event (between 0 and 1).
# Logistic regression is a classification algorithm traditionally limited to only two-class classification problems.

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd

# load the iris datasets
dataset = datasets.load_iris()
print(dataset.data.shape)


df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
print(df.corr())

# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(dataset.data, dataset.target)
print(dataset.target)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

