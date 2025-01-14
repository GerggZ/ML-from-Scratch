import numpy as np
from numpy.typing import ArrayLike
from ml_from_scratch.utils.validate_inputs import validate_k_param, validate_supervised_fit, validate_supervised_predict
from ml_from_scratch.utils.metrics import compute_pairwise_distances


class KNearestNeighbors:
    """
    A Simple K-Nearest Neighbors model made using NumPy
    """
    def __init__(self, num_neighbors: int):
        """
        Initializes necessary parameters

        Args:
            - num_neighbors (int): Number of considered nearest neighbors
        """
        self.num_neighbors = validate_k_param(num_neighbors)

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Linear Regression model to the provided training data using gradient descent

        Args:
            - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - features (ArrayLike): The input features matrix, must have shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)

        # Assign Training Data
        self.features = features
        self.targets = targets

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data (features)

        Args:
            - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features = validate_supervised_fit(features, self.targets)

        # Calculate the pairwise distances between the input data and the training data
        distances = compute_pairwise_distances(features, self.features, metric='euclidean')

        distances = np.array([np.linalg.norm(self.features - features, axis=1) for x in features])

        # Find the labels of each k-nearest neighbor for each observation in input features
        k_nearest_indices_sorted = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.labels[k_nearest_indices_sorted]

        # Use majority voting to get the most common nearest neighbor using np.bincounts()
        predictions = [np.argmax(np.bincounts(labels)) for labels in k_nearest_labels]

        return predictions


