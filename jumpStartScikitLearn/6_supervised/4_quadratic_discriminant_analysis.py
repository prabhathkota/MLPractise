# Quadratic Discriminant Analysis
from sklearn import datasets
from sklearn import metrics
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import pandas as pd

# load the iris datasets
dataset = datasets.load_iris()

print(dataset.feature_names)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
print(df.corr())

# fit a QDA model to the data
model = QDA()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

