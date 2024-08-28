"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  tree.utils import one_hot_encoding
from tree.base import DecisionTree
from metrics.py import mae,mse,accuracy,precision,recall
np.random.seed(42)
# Test case 1
# Real Input and Real Output

M = 30
P = 5
X1 = pd.DataFrame(np.random.randn(M, P))
y1 = pd.Series(np.random.randn(M))


for criteria in ["entropy", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    node = tree.fit(X1, y1,criterion=criteria)
    y_hat = tree.predict(X1,node)
    tree.plot(node)
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y1))
    print("MAE: ", mae(y_hat, y1))
print("1 done")
# # Test case 2
# # Real Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["entropy", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    node = tree.fit(X, y, criterion=criteria)
    y_hat = tree.predict(X, node)
    tree.plot(node)
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    print("Precision: ", precision(y_hat, y, average="weighted"))
    print("Recall: ", recall(y_hat, y, average="weighted"))
print("2 done")

# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")
print(one_hot_encoding(X))
for criteria in ["entropy", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    node = tree.fit(X, y, criterion=criteria)
    y_hat = tree.predict(X, node)
    tree.plot(node)
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    print("Precision: ", precision(y_hat, y, average="weighted"))
    print("Recall: ", recall(y_hat, y, average="weighted"))
print("3 done")
# # Test case 4
# # Discrete Input and Real Output
#
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["entropy", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    node = tree.fit(X, y, criterion=criteria)
    y_hat = tree.predict(X, node)
    tree.plot(node)
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))
print("4 done")
