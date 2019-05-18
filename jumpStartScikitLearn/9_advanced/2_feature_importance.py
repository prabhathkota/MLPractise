# Feature Importance
# Methods that use ensembles of decision trees (like Random Forest and Extra Trees) can also
# compute the relative importance of each attribute.
# These importance values can be used to inform a feature selection process.
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# load the iris datasets
dataset = datasets.load_iris()

# fit an Extra Trees model to the data
model = RandomForestClassifier()
model.fit(dataset.data, dataset.target)

# display the relative importance of each attribute
print(model.feature_importances_)

# Output:
# #######
# ExtraTreesClassifier
# [0.11093684 0.05503919 0.43152156 0.40250241]
#
# RandomForestClassifier
# [0.04545939 0.02104684 0.54010392 0.39338985]

