import numpy as np
from numpy.typing import ArrayLike
from ml_from_scratch.utils.validate_inputs import validate_k_param, validate_supervised_fit, validate_predict_classifier
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
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - predict_features (ArrayLike): The input predict_features matrix, must have shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)

        # Assign Training Data
        self.features = features
        self.targets = targets

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data (predict_features)

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features = validate_predict_classifier(features, self.features)

        # Calculate the pairwise distances between the input data and the training data
        distances = compute_pairwise_distances(features, self.features, metric='euclidean')

        # Find the labels of each k-nearest neighbor for each observation in input predict_features
        k_nearest_indices_sorted = np.argsort(distances, axis=1)[:, :self.num_neighbors]
        k_nearest_targets = self.targets[k_nearest_indices_sorted]

        # Use majority voting to get the most common nearest neighbor using np.bincount()
        predictions = [np.argmax(np.bincount(targets)) for targets in k_nearest_targets]

        return predictions


if __name__ == '__main__':
    print('Testing K Nearest Neighbors algorithm')
    from examples import k_nearest_neighbors
    k_nearest_neighbors(visualize=True)
