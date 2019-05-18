# Mark 0 values as missing and impute with the mean

import numpy as np
import pandas as pd
#import urllib
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

# Load the Pima Indians Diabetes dataset
#url = "http://goo.gl/j0Rvxq"
#raw_data = urllib.urlopen(url)
#dataset = np.loadtxt(raw_data, delimiter=",")

dataset = pd.read_csv('diabetes.csv')
print(dataset.columns)
print(dataset.shape)

dataset = dataset.values

# separate the data and target attributes
X = dataset[:, 0:7]
y = dataset[:, 8]
print(X)
# Mark all zero values as 0
X[X == 0] = np.nan
print(X)

# Impute all missing values with the mean of the attribute
imp = Imputer(missing_values='NaN', strategy='mean')
imputed_X = imp.fit_transform(X)

print(imputed_X)