# Linear Discriminant Analysis
# Linear Discriminate Analysis (LDA) fits a conditional probability density function (Gaussian) to each attribute
# for the class, the discrimination function between the classes is linear.

# Logistic regression is a classification algorithm traditionally limited to only two-class classification problems.

# If you have more than two classes then Linear Discriminant Analysis is the preferred linear classification technique.

from sklearn import datasets
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

# load the iris datasets
dataset = datasets.load_iris()

print(dataset.feature_names)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
print(df.corr())

# fit a LDA model to the data
model = LDA()
model.fit(dataset.data, dataset.target)
#print(model)

# make predictions
expected = dataset.target
print('-------')
print(expected)
predicted = model.predict(dataset.data)
print(predicted)
print('-------')

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))