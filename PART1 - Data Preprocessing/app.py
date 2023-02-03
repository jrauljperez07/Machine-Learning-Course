import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer


# Encoding classes
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



# Importing the dataset
dataset = pd.read_csv("Data.csv")

# Define what will be my feature variable
X = dataset.iloc[:, :-1].values
# Define my dependient variable
y = dataset.iloc[:, -1].values

# taking care of missing data consider using the average of the other rows
# to fill the empty fields
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding feature independient variable
"""
    transformers:
        1. param -> "encoder"
        2. param -> OneHotEncoder()
        3. param -> The colum/colums I wil encode, in this case I will just do it for column index [0]
"""
ct = ColumnTransformer(transformers =[('encoder',OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


print(X)