import numpy as np
from numpy.typing import ArrayLike
from ml_from_scratch.utils.nodes.tree_nodes import Node
from ml_from_scratch.utils.math import information_gain_gaussian
from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_predict_classifier


class GaussianDecisionTree:
    """
    A Gaussian Decision Tree that uses probabilistic splits based on Gaussian distributions.
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,  n_features: int | None = None):
        """
        Initializes the Gaussian Decision Tree.

        Args:
            - max_depth (int, optional): Maximum depth of the tree. If None, trees grow until other stopping criteria are met.
            - min_samples_split (int): Minimum samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Gaussian Decision Tree model to the provided training data.

        Args:
            - predict_features (ArrayLike): Feature matrix, must have shape [n_samples, n_features].
            - targets (ArrayLike): Target vector, must have shape [n_samples, ].
        """
        features, targets = validate_supervised_fit(features, targets)
        self.training_features_shape = features.shape  # stored for validating prediction inputs
        self.n_features = features.shape[1] if self.n_features is None else min(features.shape[1], self.n_features)
        self.root = self._build_tree(features, targets)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data.

        Args:
            - predict_features (ArrayLike): Feature matrix, must have shape [n_samples, n_features].

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, ].
        """
        # validate inputs and convert to numpy arrays
        features = validate_predict_classifier(features, np.zeros(self.training_features_shape))
        return np.array([self._traverse_tree(x, self.root) for x in features])

    def _build_tree(self, features: np.ndarray, targets: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds the decision tree.

        Args:
            - predict_features (np.ndarray): Feature matrix of shape [n_samples, n_features].
            - targets (np.ndarray): Target vector of shape [n_samples, ].
            - depth (int): Current depth of the tree.

        Returns:
            - Node: The root node of the tree.
        """
        num_samples, num_classes = len(targets), len(np.unique(targets))

        # Stopping conditions
        if depth == self.max_depth or num_samples < self.min_samples_split or num_classes == 1:
            return Node(value=np.bincount(targets).argmax())

        # Select random subset of predict_features if n_features is set
        feature_indices = np.random.choice(features.shape[1], self.n_features, replace=False)

        # Find the best split
        best_feature, best_mean, best_std = self._find_best_gaussian_split(features, targets, feature_indices)

        # If no valid split, return a leaf node
        if best_feature is None:
            return Node(value=np.bincount(targets).argmax())

        # Split data based on Gaussian likelihood
        left_indices, right_indices = self._gaussian_split(features[:, best_feature], best_mean, best_std)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return Node(value=np.bincount(targets).argmax())

        # Build child nodes
        left_subtree = self._build_tree(features[left_indices], targets[left_indices], depth + 1)
        right_subtree = self._build_tree(features[right_indices], targets[right_indices], depth + 1)
        return Node(feature=best_feature, mean=best_mean, std=best_std, left=left_subtree, right=right_subtree)

    def _find_best_gaussian_split(self, features: np.ndarray, targets: np.ndarray, feature_indices: list[int]):
        """
        Finds the best Gaussian split based on information gain.

        Args:
            - predict_features (np.ndarray): Feature matrix of shape [n_samples, n_features].
            - targets (np.ndarray): Target vector of shape [n_samples, ].

        Returns:
            - tuple[int | None, float | None, float | None]: Best feature index, mean, and standard deviation.
        """
        best_gain = -1
        best_feature, best_mean, best_std = None, None, None

        for feature_idx in feature_indices:
            feature_values = features[:, feature_idx]

            # Compute information gain
            gain, mean, std = information_gain_gaussian(feature_values, targets)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_mean = mean
                best_std = std

        return best_feature, best_mean, best_std

    def _gaussian_split(self, feature_column: np.ndarray, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into left and right subsets based on a threshold.

        Args:
            - feature_column (np.ndarray): A single feature column of shape [n_samples, ].
            - mean (float): The mean of the Gaussian distribution for this feature.
            - std (float): The standard deviation of the Gaussian distribution for this feature.


        Returns:
            - tuple[np.ndarray, np.ndarray]: Indices for left and right splits.
        """
        likelihoods = self._gaussian_likelihood(feature_column, mean, std)

        # Split indices based on likelihood threshold (e.g., > 0.5)
        left_indices = np.where(likelihoods > 0.5)[0]
        right_indices = np.where(likelihoods <= 0.5)[0]

        return left_indices, right_indices

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

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverses the tree to make a prediction for a single sample.

        Args:
            - x (np.ndarray): A single feature vector.
            - node (Node): The current node of the tree.

        Returns:
            - int: Predicted class label.
        """
        if node.value is not None:
            return node.value

        likelihood = self._gaussian_likelihood(x[node.feature], node.mean, node.std)

        if likelihood > 0.5:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


if __name__ == '__main__':
    print('Testing Gaussian Decision Tree algorithm')
    from examples import gaussian_decision_tree
    gaussian_decision_tree(visualize=True)
