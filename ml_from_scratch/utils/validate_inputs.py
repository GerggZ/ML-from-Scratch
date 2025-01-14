import numpy as np
from numpy.typing import ArrayLike


def validate_k_param(k: int):
    """
    Validates a k input, such as k in number of nearest neighbors

    Args:
        k (int): A k param

    Returns
        - k (int)
    """
    if not isinstance(k, int):
        raise ValueError(f"k must be an integer. Got {type(k).__name__} instead")

    if k <= 0:
        raise ValueError(f"{k} is not a valid value for k, k must be a positive integer")

    return int(k)


def validate_supervised_fit(features: ArrayLike, targets: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Validates inputs to make sure features and features are compatible for supervised training
    Converts inputs to NumPy Arrays

    Parameters:
        - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
        - features (ArrayLike): The input features matrix, must have shape [n_samples, ] or shape [n_samples, num_classes]

    Returns:
        - features, features: but converted to np.ndarrays

    Raises:
        - ValueError: If the input dimensions
    """

    # Check if inputs are actually `ArrayLike` and can be successfully converted to NumPy Arrays
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=int)

    # Validate dimensions
    if len(features.shape) != 2:
        raise ValueError(f"`features` must be a 2D array. Received {features.ndim}D array.")
    if len(targets.shape) not in [1, 2]:
        raise ValueError(f"`features` must be a 1D array (binary) or 2D array (multiclass). Received {targets.ndim}D array.")

    # Validate sample sizes
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of samples in `features` ({features.shape[0]}) and `features` ({targets.shape[0]}) must match."
        )

    return features, targets


def validate_supervised_predict(features: ArrayLike, weights: np.ndarray, bias: float | np.ndarray) -> np.ndarray:
    """
    Validates inputs to make sure features and features are compatible for supervised training
    Converts inputs to NumPy Arrays

    Parameters:
        - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
        - weights (np.ndarray): Weights of fitted model, must have shape [n_features, ]
        - bias (float | np.ndarray): Bias of fitted model

    Raises:
        - ValueError: If the input dimensions are invalid or model is not trained
    """
    # Check if inputs are actually `ArrayLike` and can be successfully converted to NumPy Arrays
    features = np.asarray(features, dtype=float)

    # Validate that weights/bias have actually been established
    if weights is None or bias is None:
        raise ValueError("Model has not been successfully trained. Either `weights` or `bias` is None")

    # Validate features dimensions
    if len(features.shape) != 2:
        raise ValueError(f"`features` must be a 2D array. Received {features.ndim}D array.")

    # Validate bias shape
    if len(weights.shape) == 1:  # Binary case
        if not np.isscalar(bias):
            raise ValueError(f"`bias` must be a scalar for binary classification. Received shape {bias.shape}.")
    elif len(weights.shape) == 2:  # Multiclass case
        if len(bias.shape) != 1 or bias.shape[0] != weights.shape[1]:
            raise ValueError(
                f"`bias` must be a 1D array of size `num_classes` ({weights.shape[1]}). Received shape {bias.shape}."
            )
    else:
        raise ValueError(f"`weights` must be 1D or 2D. Received shape {weights.shape}.")

    # Validate sample sizes
    if features.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Number of samples in `features` ({features.shape[0]}) and `weights` ({weights.shape[0]}) must match."
        )

    return features
