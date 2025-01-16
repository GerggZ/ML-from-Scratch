import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_pos_int_param, validate_predict_classifier


class PCA:
    """
    A Simple PCA implementation using NumPy
    """
    def __init__(self, n_components: int) -> None:
        """
        Initializes necessary parameters

        Args:
            - n_components (int): Number of principal components to keep
        """
        self.n_components = validate_pos_int_param(n_components)

        self.principal_components = None
        self.mean = None

    def fit(self, data: ArrayLike):
        """
        Fit PCA to the data by computing principal components.

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
        """
        # Validate input data (i.e., whether it can be converted to numpy array)
        data = np.array(data)
        self.training_data_shape = data.shape

        self.mean_vector = np.mean(data, axis=0)
        data_centered = data - self.mean_vector

        # Compute SVD (numerically stable alternative to eigen-decomposition)
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

        # Store the top `n_components` principal components
        self.principal_components = Vt[:self.n_components]

    def transform(self, data: ArrayLike) -> np.ndarray:
        """
        Projects data into the reduced dimensional space

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Projected data, shape [m_samples, n_components]
        """
        data = validate_predict_classifier(data, self.training_data_shape)
        data_centered = data - self.mean_vector
        return np.dot(data_centered, self.components.T)

    def inverse_transform(self, data: ArrayLike) -> np.ndarray:
        """
        Re-projects data into the expanded, or original, dimensional space

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Projected data, shape [m_samples, n_features]
        """
        return np.dot(data, self.principal_components) + self.mean_vector

