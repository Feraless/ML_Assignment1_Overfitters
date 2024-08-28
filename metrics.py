from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy.
    Accuracy = (Number of correct predictions) / (Total number of predictions)
    """
    # Check if the sizes of y_hat and y are equal
    assert y_hat.size == y.size, "Size of predicted and actual values must be equal."
    # Ensure that y_hat and y are not empty
    assert y_hat.size > 0, "Predicted and actual values cannot be empty."

    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    accuracy_value = correct_predictions / total_predictions
    return accuracy_value


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision.
    Precision = (True Positives) / (True Positives + False Positives)
    """
    # Ensure that y_hat and y are not empty
    assert y_hat.size > 0, "Predicted and actual values cannot be empty."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()

    # Avoid division by zero
    if (true_positives + false_positives) == 0:
        return 0.0

    precision_value = true_positives / (true_positives + false_positives)
    return precision_value


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall.
    Recall = (True Positives) / (True Positives + False Negatives)
    """
    # Ensure that y_hat and y are not empty
    assert y_hat.size > 0, "Predicted and actual values cannot be empty."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()

    # Avoid division by zero
    if (true_positives + false_negatives) == 0:
        return 0.0

    recall_value = true_positives / (true_positives + false_negatives)
    return recall_value


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (RMSE).
    RMSE = sqrt(mean((y_hat - y)^2))
    """
    # Check if the sizes of y_hat and y are equal
    assert y_hat.size == y.size, "Size of predicted and actual values must be equal."
    # Ensure that y_hat and y are not empty
    assert y_hat.size > 0, "Predicted and actual values cannot be empty."

    mse = np.mean((y_hat - y) ** 2)
    rmse_value = np.sqrt(mse)
    return rmse_value


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (MAE).
    MAE = mean(abs(y_hat - y))
    """
    # Check if the sizes of y_hat and y are equal
    assert y_hat.size == y.size, "Size of predicted and actual values must be equal."
    # Ensure that y_hat and y are not empty
    assert y_hat.size > 0, "Predicted and actual values cannot be empty."

    mae_value = np.mean(np.abs(y_hat - y))
    return mae_value
