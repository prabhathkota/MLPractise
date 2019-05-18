# Load the Pima Indians diabetes dataset from CSV
import numpy as np
import pandas as pd
import urllib.request

# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://goo.gl/j0Rvxq"
# download the file
#raw_data = urllib.request.urlopen(url)


# load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")

dataset = pd.read_csv('diabetes.csv')
#print(dataset.shape)
#print(dataset)
print(type(dataset)) #<class 'pandas.core.frame.DataFrame'>
print(dataset.head())
print(dataset.corr())

dataset = dataset.values
print(type(dataset))  #<class 'numpy.ndarray'>

# separate the data from the target attributes
X = dataset[:, 0:7]
#print(X)
y = dataset[:, 8]
#print(y)
