# Recursive Feature Elimination
# The Recursive Feature Elimination (RFE) method is a feature selection approach.
# It works by recursively removing attributes and building a model on those attributes that remain.
# It uses the model accuracy to identify which attributes (and combination of attributes) contribute the
# most to predicting the target attribute.
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

# load the iris datasets
dataset = datasets.load_iris()

print(dataset.feature_names)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
print(df.corr())

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()

# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)



