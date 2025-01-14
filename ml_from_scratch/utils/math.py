import numpy as np

def entropy(targets: np.ndarray) -> float:
    """
    Computes the entropy of a target distribution

    Args:
        - features (np.ndarray): Target values (e.g., class labels)
    """

    class_counts = np.bincount(targets)
    probabilities = class_counts / len(targets)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy


def information_gain(feature_column: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    """
    Computes the information gain for a potential split.

    Args:
        - feature_column (np.ndarray): Feature values of shape [n_samples, ].
        - features (np.ndarray): Target values of shape [n_samples, ].
        - threshold (float): Threshold for splitting.

    Returns:
        float: Information gain from the split.
    """
    parent_entropy = entropy(targets)

    left_indices = feature_column <= threshold
    right_indices = ~left_indices

    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    left_child_entropy = entropy(targets[left_indices])
    right_child_entropy = entropy(targets[right_indices])
    left_weight = len(left_indices) / len(feature_column)
    right_weight = len(right_indices) / len(feature_column)

    information_gain = parent_entropy - (left_weight * left_child_entropy + right_weight * right_child_entropy)
    return information_gain
