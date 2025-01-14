import numpy as np
from numpy.typing import ArrayLike
from ml_from_scratch.utils.nodes.tree_nodes import Node
from ml_from_scratch.utils.math import information_gain
from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_supervised_predict


class GaussianDecisionTree:
    """
    A Gaussian Decision Tree that uses probabilistic splits based on Gaussian distributions.
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2):
        """
        Initializes the Gaussian Decision Tree.

        Args:
            - max_depth (int, optional): Maximum depth of the tree. If None, trees grow until other stopping criteria are met.
            - min_samples_split (int): Minimum samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Gaussian Decision Tree model to the provided training data.

        Args:
            - features (ArrayLike): Feature matrix, must have shape [n_samples, n_features].
            - targets (ArrayLike): Target vector, must have shape [n_samples, ].
        """
        features, targets = validate_supervised_fit(features, targets)
        self.root = self._build_tree(features, targets)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data.

        Args:
            - features (ArrayLike): Feature matrix, must have shape [n_samples, n_features].

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, ].
        """
        # validate inputs and convert to numpy arrays
        features = validate_supervised_predict(features, self.weights, self.bias)

        return np.array([self._traverse_tree(x, self.root) for x in features])

    def _build_tree(self, features: np.ndarray, targets: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds the decision tree.

        Args:
            - features (np.ndarray): Feature matrix of shape [n_samples, n_features].
            - targets (np.ndarray): Target vector of shape [n_samples, ].
            - depth (int): Current depth of the tree.

        Returns:
            - Node: The root node of the tree.
        """
        num_samples, num_features = features.shape
        num_classes = len(np.unique(targets))

        # Stopping conditions
        if depth == self.max_depth or num_samples < self.min_samples_split or num_classes == 1:
            return Node(value=self._majority_class(targets))

        # Find the best split
        best_feature, best_mean, best_std = self._find_best_gaussian_split(features, targets)

        # If no valid split, return a leaf node
        if best_feature is None:
            return Node(value=self._majority_class(targets))

        # Split data based on Gaussian likelihood
        left_indices = self._gaussian_likelihood(features[:, best_feature], best_mean, best_std) > 0.5
        right_indices = ~left_indices

        # Build child nodes
        left_subtree = self._build_tree(features[left_indices], targets[left_indices], depth + 1)
        right_subtree = self._build_tree(features[right_indices], targets[right_indices], depth + 1)

        return Node(feature=best_feature, mean=best_mean, std=best_std, left=left_subtree, right=right_subtree)

    def _find_best_gaussian_split(self, features: np.ndarray, targets: np.ndarray):
        """
        Finds the best Gaussian split based on information gain.

        Args:
            - features (np.ndarray): Feature matrix of shape [n_samples, n_features].
            - targets (np.ndarray): Target vector of shape [n_samples, ].

        Returns:
            - tuple[int | None, float | None, float | None]: Best feature index, mean, and standard deviation.
        """
        best_gain = -1
        best_feature, best_mean, best_std = None, None, None

        for feature_idx in range(features.shape[1]):
            feature_values = features[:, feature_idx]

            # Compute Gaussian parameters
            mean, std = np.mean(feature_values), np.std(feature_values)

            # Compute likelihood scores
            likelihoods = self._gaussian_likelihood(feature_values, mean, std)

            # Compute information gain
            gain = information_gain(targets, likelihoods)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_mean = mean
                best_std = std

        return best_feature, best_mean, best_std

    @staticmethod
    def _gaussian_likelihood(values: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Computes the Gaussian likelihood for a set of values.

        Args:
            - values (np.ndarray): Feature values of shape [n_samples, ].
            - mean (float): Mean of the Gaussian distribution.
            - std (float): Standard deviation of the Gaussian distribution.

        Returns:
            - np.ndarray: Gaussian likelihoods of shape [n_samples, ].
        """
        return np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    @staticmethod
    def _majority_class(targets: np.ndarray) -> int:
        """
        Finds the most common class label in the targets.

        Args:
            - targets (np.ndarray): Target vector of shape [n_samples, ].

        Returns:
            - int: The most common class label.
        """
        return np.bincount(targets).argmax()

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverses the tree to make a prediction for a single sample.

        Args:
            - x (np.ndarray): A single feature vector.
            - node (Node): The current node of the tree.

        Returns:
            - int: Predicted class label.
        """
        if node.is_leaf_node():
            return node.value

        likelihood = self._gaussian_likelihood(x[node.feature], node.mean, node.std)

        if likelihood > 0.5:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
