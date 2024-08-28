"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
from turtledemo.sorting_animate import partition
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
import numpy as np
from metrics.py import mae,rmse,accuracy,precision,recall
from sklearn.metrics import mean_squared_error


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    X_encoded = pd.get_dummies(X)
    X_encoded.replace({False: 0, True: 1}, inplace=True)
    return X_encoded


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype== "category":
        return False
    if pd.api.types.is_integer_dtype(y):
        return False
    elif pd.api.types.is_float_dtype(y):
        if y.apply(float.is_integer).all():
            return False
        else:
            return True
    else:
        raise ValueError("Series must contain numeric values.")


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    sum_entropy = 0
    for count in Y.value_counts():
        prob = count / len(Y)
        sum_entropy -= prob * np.log2(prob + 1e-10)
    return sum_entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    proportions = Y.value_counts(normalize=True)
    return 1.0 - np.sum(proportions ** 2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    if criterion == "entropy":
        initial = entropy(Y)
        weighted_entropy = 0.0
        for value in attr.unique():
            subset = Y[attr == value]
            weighted_entropy += (len(subset) / len(Y)) * entropy(subset)
        return initial - weighted_entropy

    elif criterion == "gini_index":
        initial = gini_index(Y)
        weighted_gini = 0.0
        for value in attr.unique():
            subset = Y[attr == value]
            weighted_gini += (len(subset) / len(Y)) * gini_index(subset)
        return initial - weighted_gini


    elif criterion == "mse":
        initial = np.mean((Y - np.mean(Y)) ** 2)
        weighted_mse = 0.0
        for value in attr.unique():
            subset = Y[attr == value]
            weighted_mse += (len(subset) / len(Y)) * np.mean((subset - np.mean(subset)) ** 2)
        return initial - weighted_mse
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features,criterion = None):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    best_gain = -float('inf')
    best_attr = None
    best_split_value = None

    for attr in features:
        gain = 0.0
        if check_ifreal(X[attr]):
            if check_ifreal(y):
                a, b, c, d, e, min_impurity = split_data(X, y,attr)
                gain = mean_squared_error(y,[y.mean()]*len(y)) - min_impurity
            else:
                a, b, c, d, e, min_impurity = split_data(X, y,attr)
                if criterion == "entropy":
                    gain = entropy(y) - min_impurity
                elif criterion == "gini_index":
                    gain = gini_index(y) - min_impurity
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        else:
            if check_ifreal(y):
                gain = information_gain(y, X[attr], "mse")
            else:
                gain = information_gain(y, X[attr], criterion=criterion)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
    return best_attr
def split_data(X: pd.DataFrame, y: pd.Series, attribute, value = None):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: split data(Input and output)
    """
    if not check_ifreal(X[attribute]):
        opt_value = None
        X[attribute] = X[attribute].astype(int)
        left = X[attribute] <= value
        right = X[attribute] > value
        min_impurity = None
        X_left = X[left]
        X_right = X[right]
        y_left = y[left]
        y_right = y[right]

    else:
        # Real-valued feature split
        df = pd.DataFrame({"attr": X[attribute], "y": y})
        df.sort_values(by='attr', inplace=True)

        min_impurity = np.inf
        opt_value = None

        # Calculate impurity for each possible split
        for i in range(1, len(df)):
            split_value = (df['attr'].iloc[i] + df['attr'].iloc[i - 1]) / 2
            left_y = df[df['attr'] <= split_value]['y']
            right_y = df[df['attr'] > split_value]['y']
            count_left = len(left_y)
            count_right = len(right_y)
            total_count = count_left + count_right

            if check_ifreal(left_y):
                # MSE for regression
                if count_left > 0:
                    left_impurity = mean_squared_error(left_y, [left_y.mean()] * len(left_y))
                else:
                    left_impurity = 0
                if count_right > 0:
                    right_impurity = mean_squared_error(right_y, [right_y.mean()] * len(right_y))
                else:
                    right_impurity = 0
            else:
                # Entropy for classification
                if count_left > 0:
                    left_impurity = entropy(left_y)
                else:
                    left_impurity = 0
                if count_right > 0:
                    right_impurity = entropy(right_y)
                else:
                    right_impurity = 0

            total_impurity = left_impurity * (count_left / total_count) + right_impurity * (count_right / total_count)
            if total_impurity < min_impurity:
                min_impurity = total_impurity
                opt_value = split_value

        # Apply the best split found
        left = X[attribute] <= opt_value
        right = X[attribute] > opt_value

        X_left = X[left]
        X_right = X[right]
        y_left = y[left]
        y_right = y[right]

    return X_left, y_left, X_right, y_right, opt_value,min_impurity
