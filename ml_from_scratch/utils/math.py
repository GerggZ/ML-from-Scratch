import numpy as np


def entropy(targets: np.ndarray) -> float:
    """
    Computes the entropy of a target distribution

    Args:
        - predict_data (np.ndarray): Target values (e.g., class labels)
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
        - predict_data (np.ndarray): Target values of shape [n_samples, ].
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


def information_gain_gaussian(feature_column: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the information gain for a Gaussian split.

    Args:
        - feature_column (np.ndarray): Feature values of shape [n_samples, ].
        - targets (np.ndarray): Target values of shape [n_samples, ].

    Returns:
        float: Information gain from the Gaussian split.
    """
    # Compute mean and standard deviation of the feature column
    mean = np.mean(feature_column)
    std = np.std(feature_column)

    # Skip if the standard deviation is zero
    if std == 0:
        return 0, 0, 0

    # Compute Gaussian likelihoods
    likelihoods = np.exp(-0.5 * ((feature_column - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    likelihoods = (likelihoods - np.min(likelihoods)) / (np.max(likelihoods) - np.min(likelihoods))

    # Calculate parent entropy
    parent_entropy = entropy(targets)

    # Split targets based on Gaussian likelihood > 0.5
    left_indices = likelihoods > 0.5
    right_indices = ~left_indices

    if np.sum(left_indices) == 0:
        j = 7
    if np.sum(right_indices) == 0:
        j = 87

    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
        return 0, 0, 0

    # Calculate child entropies
    left_entropy = entropy(targets[left_indices])
    right_entropy = entropy(targets[right_indices])

    # Calculate information gain
    left_weight = len(left_indices) / len(feature_column)
    right_weight = len(right_indices) / len(feature_column)
    child_entropy = left_weight * left_entropy + right_weight * right_entropy

    return parent_entropy - child_entropy, mean, std
