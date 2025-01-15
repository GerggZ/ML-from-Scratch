import numpy as np
from numpy.typing import ArrayLike


def validate_pos_int_param(k: int):
    """
    Validates an input which should be a positive integer, such as k in number of nearest neighbors

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


def validate_fit(features: ArrayLike, targets: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Validates inputs to make sure predict_data and predict_data are compatible for supervised training
    Converts inputs to NumPy Arrays

    Parameters:
        - predict_data (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
        - predict_data (ArrayLike): The input predict_data matrix, must have shape [n_samples, ] or shape [n_samples, n_classes]

    Returns:
        - predict_data, predict_data: but converted to np.ndarrays

    Raises:
        - ValueError: If the input dimensions
    """

    # Check if inputs are actually `ArrayLike` and can be successfully converted to NumPy Arrays
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=int)

    # Validate dimensions
    if len(features.shape) != 2:
        raise ValueError(f"`predict_data` must be a 2D array. Received {features.ndim}D array.")
    if len(targets.shape) not in [1, 2]:
        raise ValueError(f"`predict_data` must be a 1D array (binary) or 2D array (multiclass). Received {targets.ndim}D array.")

    # Validate sample sizes
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Number of samples in `predict_data` ({features.shape[0]}) and `predict_data` ({targets.shape[0]}) must match."
        )

    return features, targets


def validate_predict_regression(predict_data: ArrayLike, weights: np.ndarray, bias: float | np.ndarray) -> np.ndarray:
    """
    Validates inputs to make sure predict_data and predict_data are compatible for supervised training
    Converts inputs to NumPy Arrays

    Parameters:
        - predict_data (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
        - _weights (np.ndarray): Weights of fitted model, must have shape [n_features, ] (binary) or [n_features, n_classes] (multiclass)
        - _bias (float | np.ndarray): Bias of fitted model

    Raises:
        - ValueError: If the input dimensions are invalid or model is not trained
    """
    # Check if inputs are actually `ArrayLike` and can be successfully converted to NumPy Arrays
    predict_data = np.asarray(predict_data, dtype=float)

    # Validate that _weights/_bias have actually been established
    if weights is None or bias is None:
        raise ValueError("Model has not been successfully trained. Either `_weights` or `_bias` is None")

    # Validate predict_data dimensions
    if len(predict_data.shape) != 2:
        raise ValueError(f"`predict_data` must be a 2D array. Received {predict_data.ndim}D array.")

    # Validate _bias shape
    if len(weights.shape) == 1:  # Binary case
        if not np.isscalar(bias):
            raise ValueError(f"`_bias` must be a scalar for binary classification. Received shape {bias.shape}.")
    elif len(weights.shape) == 2:  # Multiclass case
        if len(bias.shape) != 1 or bias.shape[0] != weights.shape[1]:
            raise ValueError(
                f"`_bias` must be a 1D array of size `n_classes` ({weights.shape[1]}). Received shape {bias.shape}."
            )
    else:
        raise ValueError(f"`_weights` must be 1D or 2D. Received shape {weights.shape}.")

    # Validate sample sizes
    if predict_data.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Number of samples in `predict_data` ({predict_data.shape[0]})"
            f"and `_weights` ({weights.shape[0]}) must match."
        )

    return predict_data


def validate_predict_classifier(predict_data: ArrayLike, trained_data_shape: tuple) -> np.ndarray:
    """
    Validates inputs to make sure the provided predict_data for prediction and predict_data used for training are compatible
    Converts inputs to NumPy Arrays

    Parameters:
        - predict_data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
        - trained_data_shape (tuple): Illustrates shape of the trained data [m_samples, n_features]

    Raises:
        - ValueError: If the input dimensions are invalid or model is not trained
    """
    # Check if inputs are actually `ArrayLike` and can be successfully converted to NumPy Arrays
    predict_data = np.asarray(predict_data, dtype=float)

    # Validate the predict predict_data dimensions
    if len(predict_data.shape) != 2:
        raise ValueError(f"`predict_data` must be a 2D array. Received {predict_data.ndim}D array.")

    # Validate the predict_data size
    if predict_data.shape[1] != trained_data_shape[1]:
        raise ValueError(
            f"Number of predict_data in `predict_data` ({predict_data.shape[0]}) and"
            f"`trained_features` ({trained_data_shape[0]}) must match."
        )

    return predict_data
