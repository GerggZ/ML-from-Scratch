import numpy as np
from numpy.typing import ArrayLike


def mean_squared_error(actual: ArrayLike, predicted: ArrayLike) -> float:
    """
    Computes the mean squared error (MSE) between the actual and predicted values

    Args:
        - actual (ArrayLike): Ground truth target values
        - predicted (ArrayLike): Predicted target values

    Returns:
        - float: the mean squared error
    """
    return np.mean((np.array(actual) - np.array(predicted)) ** 2)


def mean_absolute_error(actual: ArrayLike, predicted: ArrayLike) -> float:
    """
    Computes the mean absolute error (MSE) between the actual and predicted values

    Args:
        - actual (ArrayLike): Ground truth target values
        - predicted (ArrayLike): Predicted target values

    Returns:
        - float: the mean absolute error
    """
    return np.mean(np.abs((np.array(actual) - np.array(predicted))))


def r_squared_score(actual: ArrayLike, predicted: ArrayLike) -> float:
    """
    Computes the R-squared (coefficient of determination) score between the actual nd predicted values

    Args:
        - actual (ArrayLike): Ground truth target values
        - predicted (ArrayLike): Predicted target values

    Returns:
        - float: the r-squared (coefficient of determination) score
    """
    return np.mean((np.array(actual) - np.array(predicted)) ** 2)
