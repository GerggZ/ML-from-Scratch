import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.algorithms.supervised.decision_tree import DecisionTree
from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_supervised_predict


class RandomForest:
    """
    A Simple Random Forest model made using NumPy
    """

    def __init__(self, n_trees: int, min_samples_split: int = 2, max_depth: int | None = 10, num_features: int | None = None) -> None:
        """
        Initializes necessary parameters

        Args:
            - n_trees (int): Number of decision trees in the forest
            - min_samples_split (int): Minimum samples required to split a node
            - max_depth (int | None): Maximum depth of each tree. If None, trees grow until other stopping criteria are met
            - n_features (int | None): Number of features to randomly select at each split. If None, all features are used
        """

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features

        self.trees: list[DecisionTree] = []

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Decision Tree model to the provided training data using gradient descent

        Args:
            - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input features matrix, must have shape [n_samples, ]
        """
        # validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)
        self.trees = []

        for _ in range(self.n_trees):
            # Create a new decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, num_features=self.n_features)

            # Generate bootstrap samples
            features_subset, targets_subset = self._bootstrap_samples(features, targets)

            # Fit the tree on the bootstrap sample
            tree.fit(features_subset, targets_subset)

            # Add the trained tree to the forest
            self.trees.append(tree)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data (features)

        Args:
            - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, num_classes]
        """
        # validate inputs and convert to numpy arrays
        features = validate_supervised_predict(features, self.weights, self.bias)

        # Collect predictions from all trees in the forest
        tree_predictions = np.array([tree.predict(features) for tree in self.trees])

        # Majority vote for each sample
        predictions = np.apply_along_axis(lambda labels: np.bincount(labels).argmax(), axis=0, arr=tree_predictions)

        return predictions

    @staticmethod
    def _bootstrap_samples(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a bootstrap sample from the data.

        Args:
            - features (np.ndarray): Feature matrix of shape [n_samples, n_features].
            - targets (np.ndarray): Target vector of shape [n_samples, ].

        Returns:
            - tuple[np.ndarray, np.ndarray]: Bootstrap feature matrix and target vector.
        """
        n_samples = features.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return features[idxs], targets[idxs]