import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics.py import mae,mse,accuracy,precision,recall
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

# Part a)

X = pd.DataFrame(X)
y = pd.Series(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from tree.base import DecisionTree
tree = DecisionTree(criterion="entropy")
node = tree.fit(X_train, y_train,criterion="entropy")
y_hat = tree.predict(X_test,node)
print("Accuracy: ", accuracy(y_hat, y_test))
print("Precision: ", precision(y_hat, y_test,average="weighted"))
print("Recall: ", recall(y_hat, y_test,average="weighted"))

# Part b)

tree = DecisionTree(criterion="entropy")
best_acc = 0
best_depth = 0
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X.iloc[indices].reset_index(drop=True)
y = y.iloc[indices].reset_index(drop=True)
folds = np.array_split(indices, 5)
for depth in range(5):
    acc = 0
    for i in range(5):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(5) if j != i])
        train, val = X.iloc[train_indices], X.iloc[val_indices]
        train_y, val_y = y.iloc[train_indices], y.iloc[val_indices]
        node = tree.fit(train, train_y, criterion="entropy")
        y_hat = tree.predict(val, node)
        acc += accuracy(y_hat, val_y)
    acc = acc / 5
    if acc > best_acc:
        acc = best_acc
        best_depth = depth
print(best_depth)

