import numpy as np


def accuracy_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute accuracy score.

    Args:
        actual (np.ndarray): True labels.
        predicted (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(actual == predicted)


def precision_score(actual: np.ndarray, predicted: np.ndarray, positive_label: int = 1) -> float:
    """
    Compute precision score.

    Args:
        actual (np.ndarray): True labels.
        predicted (np.ndarray): Predicted labels.
        positive_label (int): The label considered as positive.

    Returns:
        float: Precision score.
    """
    tp = np.sum((predicted == positive_label) & (actual == positive_label))
    fp = np.sum((predicted == positive_label) & (actual != positive_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(actual: np.ndarray, predicted: np.ndarray, positive_label: int = 1) -> float:
    """
    Compute recall score.

    Args:
        actual (np.ndarray): True labels.
        predicted (np.ndarray): Predicted labels.
        positive_label (int): The label considered as positive.

    Returns:
        float: Recall score.
    """
    tp = np.sum((predicted == positive_label) & (actual == positive_label))
    fn = np.sum((predicted != positive_label) & (actual == positive_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(actual: np.ndarray, predicted: np.ndarray, positive_label: int = 1) -> float:
    """
    Compute F1-score.

    Args:
        actual (np.ndarray): True labels.
        predicted (np.ndarray): Predicted labels.
        positive_label (int): The label considered as positive.

    Returns:
        float: F1-score.
    """
    precision = precision_score(actual, predicted, positive_label)
    recall = recall_score(actual, predicted, positive_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
