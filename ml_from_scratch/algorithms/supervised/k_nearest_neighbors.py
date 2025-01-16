import numpy as np
from numpy.typing import ArrayLike
from ml_from_scratch.utils.validate_inputs import validate_pos_int_param, validate_fit, validate_predict_classifier
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
        self.num_neighbors = validate_pos_int_param(num_neighbors)

        self.data = None
        self.targets = None
        self.training_data_shape = None


    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the K Nearest Neighbors algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ]
        """
        self.data, self.targets = validate_fit(data, targets)
        self.training_data_shape = data.shape

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ]
        """
        data = validate_predict_classifier(data, self.training_data_shape)

        # Calculate the pairwise distances between the input data and the training data
        distances = compute_pairwise_distances(data, self.data, metric='euclidean')

        # Find the labels of each k-nearest neighbor for each observation in input predict_data
        k_nearest_indices_sorted = np.argsort(distances, axis=1)[:, :self.num_neighbors]
        k_nearest_targets = self.targets[k_nearest_indices_sorted]

        # Use majority voting to get the most common nearest neighbor using np.bincount()
        predictions = [np.argmax(np.bincount(targets)) for targets in k_nearest_targets]

        return predictions


if __name__ == '__main__':
    print('Testing K Nearest Neighbors algorithm')
    from examples import k_nearest_neighbors_example
    k_nearest_neighbors_example(visualize=True)
