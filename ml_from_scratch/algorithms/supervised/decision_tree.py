import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.nodes.tree_nodes import Node
from ml_from_scratch.utils.math import information_gain
from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_supervised_predict


class DecisionTree:
    """
    A Simple Decision Tree model made using NumPy
    """
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,  num_features: int | None = None) -> None:
        """
        Initializes necessary parameters

        Args:
            - max_depth (int, optional): The maximum depth of the tree. Defaults to None
            - min_samples_split (int): The minimum number of samples required to split a node. Defaults to 2
            - num_features (int, optional): A number of eatures to randomly consider at each split (default None uses all features)
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features

        self.root = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Decision Tree model to the provided training data using gradient descent

        Args:
            - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input features matrix, must have shape [n_samples, ]
        """
        # validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)

        self.root = self._build_tree(features, targets)

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

        return np.array([self._traverse_tree(x, self.root) for x in features])

    def _build_tree(self, features: np.ndarray, targets: np.ndarray, depth: int = 0):
        """
        Recursively builds the decision tree

        Args:
            - features (np.ndarray): The input feature matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input features matrix, must have shape [n_samples, ]
            - depth (int): Current depth of tree

        Returns:
            - Node: Recursively this would provide the root node!
        """
        num_samples, num_features = features.shape
        num_classes = len(np.unique(targets))

        # Stopping conditions
        if depth == self.max_depth or num_samples < self.min_samples_split or num_classes == 1:
            return Node(value=self._majority_class(targets))

        # Select random subset of features if num_features is set
        feature_indices = np.arange(num_features)
        if self.num_features is not None:
            feature_indices = np.random.choice(num_features, self.num_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(features, targets, feature_indices)

        # If no valid split, return a leaf node
        if best_feature is None:
            return Node(value=self._majority_class(targets))

        # Split the data
        left_indices = features[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Build child nodes
        left_subtree = self._build_tree(features[left_indices], targets[left_indices], depth + 1)
        right_subtree = self._build_tree(features[right_indices], targets[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    @staticmethod
    def _best_split(features, targets, feat_idxs):
        """
        Finds the best feature and threshold to split on

        Args:
            - features (np.ndarray): Feature matrix of shape [n_samples, n_features].
            - targets (np.ndarray): Target vector of shape [n_samples, ].
            - feature_indices (np.ndarray): Indices of features to consider for the split.

        Returns:
            - tuple[int | None, float | None]: Best feature index and best threshold for splitting, or (None, None) if no valid split is found.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            feature_column = features[:, feat_idx]
            thresholds = np.unique(feature_column)

            for threshold in thresholds:
                # Calculate the information gain
                gain = information_gain(feature_column, targets, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_threshold = feat_idx, threshold

        return split_idx, split_threshold

    @staticmethod
    def _split(feature_column: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into left and right subsets based on a threshold.

        Args:
            - feature_column (np.ndarray): A single feature column of shape [n_samples, ].
            - threshold (float): The threshold value for splitting.

        Returns:
            - tuple[np.ndarray, np.ndarray]: Indices for left and right splits.
        """
        left_indices = np.where(feature_column <= threshold)[0]
        right_indices = np.where(feature_column > threshold)[0]
        return left_indices, right_indices

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
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)