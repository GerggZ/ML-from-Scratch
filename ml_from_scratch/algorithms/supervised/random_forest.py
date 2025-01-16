import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.algorithms.supervised.decision_tree import DecisionTree
from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_classifier


class RandomForest:
    """
    A Simple Random Forest model made using NumPy
    """

    def __init__(self, n_trees: int, min_samples_split: int = 2, max_depth: int | None = 10, n_features: int | None = None) -> None:
        """
        Initializes necessary parameters

        Args:
            - n_trees (int): Number of decision _trees in the forest
            - min_samples_split (int): Minimum samples required to split a node
        Optional Args:
            - max_depth (int): Maximum depth of each tree
                               If None, the trees grow until other stopping criteria are met
            - n_features (int): Number of predict_data to randomly select at each split
                                If None, all predict_data are used
        """

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self._trees: list[DecisionTree] = []

        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Random Forest algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ]
        """
        data, targets = validate_fit(data, targets)
        self.training_data_shape = data.shape

        self._trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features
            )

            data_subset, targets_subset = self._bootstrap_samples(data, targets)
            tree.fit(data_subset, targets_subset)

            # Add the trained tree to the forest
            self._trees.append(tree)

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ]
        """
        data = validate_predict_classifier(data, self.training_data_shape)

        # Collect predictions from all _trees in the forest
        tree_predictions = np.array([tree.predict(data) for tree in self._trees])

        # Majority vote for each sample
        predictions = np.apply_along_axis(lambda labels: np.bincount(labels).argmax(), axis=0, arr=tree_predictions)

        return predictions

    @staticmethod
    def _bootstrap_samples(data: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a bootstrap sample from the data.

        Args:
            - data (np.ndarray): The input data matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input target matrix, must have shape [n_samples, ]

        Returns:
            - tuple: Subset of the data and targets
        """
        n_samples = data.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return data[idxs], targets[idxs]


if __name__ == '__main__':
    print('Testing Random Forest algorithm')
    from examples import random_forest_example
    random_forest_example(visualize=True)
