"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Any, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error,root_mean_squared_error as rmse, accuracy_score, mean_absolute_error as mae,precision_score
from metrics.py import mae,rmse,accuracy,precision,recall
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import check_ifreal, split_data, one_hot_encoding, opt_split_attribute
import graphviz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
np.random.seed(42)


@dataclass
class Node:
    attribute: str = None
    threshold: Any = None
    left: Any = None
    right: Any = None
    value: int = None
    def __init__(self, left = None, right = None, attribute=None, threshold=None, value = None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

@dataclass
class DecisionTree:
    criterion: Literal["entropy", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.node = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series,  criterion: str, depth=0, counter=0) -> dict[str, Any] | Any:

        if counter == 0:
            X = one_hot_encoding(X)

        if y.empty:
            return Node(value=None)

        if len(set(y)) == 1:
            return Node(value=y.iloc[0])

        if depth >= self.max_depth:
            mode_value = y.mode()
            return Node(value=mode_value[0])

        if len(X.columns) == 0:
            mode_value = y.mode()
            return Node(value=mode_value[0])

        attr = opt_split_attribute(X, y, X.columns, self.criterion)

        if attr is None:
            mode_value = y.mode()
            return Node(value=mode_value[0])

        if not check_ifreal(X[attr]):
            X_attr = X[attr].astype(int)
            threshold = X_attr.mean()
            X_left, y_left, X_right, y_right,_,nul = split_data(X, y, attr, threshold)
        else:
            X_left, y_left, X_right, y_right,threshold,_ = split_data(X, y, attr)

        left_subtree = self.fit(X_left, y_left, criterion=criterion, depth=depth + 1, counter=counter + 1)
        right_subtree = self.fit(X_right, y_right, criterion=criterion, depth=depth + 1, counter=counter + 1)

        return Node(attribute=attr, threshold=threshold, left=left_subtree, right=right_subtree)


    def predict(self, X: pd.DataFrame,node):
        """
        Function to run the decision tree on test inputs
        """
        X = one_hot_encoding(X)
        return X.apply(self.predict_single,node = node,axis=1)

    def plot(self,node) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        def add_edges(node, graph, parent_name=None):
            if node is None or (node.attribute is None and node.threshold is None):
                return

            if node.value is not None:
                node_name = f'leaf_{id(node)}'
                graph.node(node_name, label=str(node.value), shape='box', style='filled', color='lightgrey')
                if parent_name:
                    graph.edge(parent_name, node_name)
                return

            node_name = f'node_{id(node)}'
            label = f"{node.attribute} < {node.threshold}"
            graph.node(node_name, label=label)

            if parent_name:
                graph.edge(parent_name, node_name)

            add_edges(node.left, graph, node_name)
            add_edges(node.right, graph, node_name)

        dot = graphviz.Digraph()
        add_edges(node, dot)
        dot.render("decision_tree", format="png",view = True)

    def predict_single(self, x, node):
        if node is None:
            node = self.node  # Start from the root if no specific node is given

        if node.value is not None:
            return node.value

        if x[node.attribute] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

