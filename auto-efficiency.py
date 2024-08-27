import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from tree.base import DecisionTree
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

data.replace('?', np.nan, inplace=True)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['mpg'] = pd.to_numeric(data['mpg'], errors='coerce')
data['displacement'] = pd.to_numeric(data['displacement'], errors='coerce')
data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
data['acceleration'] = pd.to_numeric(data['acceleration'], errors='coerce')
data['cylinders'] = pd.to_numeric(data['cylinders'], errors='coerce')
data['model year'] = pd.to_numeric(data['model year'], errors='coerce')
data['origin'] = pd.to_numeric(data['origin'], errors='coerce')
data.dropna(inplace=True)
data['origin'] = data['origin'].astype('category')
data.drop(["car name","cylinders"], axis=1, inplace=True) #Obsolete feautures car name and cylinders
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop("mpg",axis =1), data["mpg"], test_size=0.3, random_state=1)
tree = DecisionTree(criterion="entropy",max_depth=4)
node = tree.fit(X_train, y_train,criterion="entropy")
y_hat = tree.predict(X_test,node)
print("My Decision Tree MSE:",mse(y_test,y_hat))

# Sklearn Decision Tree
tr = DecisionTreeRegressor()
tr.fit(X_train, y_train)
y_hat = tr.predict(X_test)
print("Sklearn Decision Tree MSE:",mse(y_test,y_hat))

