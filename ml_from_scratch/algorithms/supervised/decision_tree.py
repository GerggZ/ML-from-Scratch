import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.nodes.tree_nodes import Node
from ml_from_scratch.utils.math import information_gain
from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_classifier


class DecisionTree:
    """
    A Simple Decision Tree model made using NumPy
    """
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,  n_features: int | None = None) -> None:
        """
        Initializes necessary parameters

        Args:
            - max_depth (int, optional): The maximum depth of the tree. Defaults to None
            - min_samples_split (int): The minimum number of samples required to split a node. Defaults to 2
            - n_features (int, optional): A number of data to randomly consider at each split (None uses all features)
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

        self._root = None
        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Decision Tree algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ]
        """
        # validate inputs and convert to numpy arrays
        data, targets = validate_fit(data, targets)
        self.training_data_shape = data.shape

        self.n_features = data.shape[1] if self.n_features is None else min(data.shape[1], self.n_features)
        self._root = self._build_tree(data, targets)

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ]
        """
        # validate inputs and convert to numpy arrays
        data = validate_predict_classifier(data, self.training_data_shape)
        return np.array([self._traverse_tree(data_sample, self._root) for data_sample in data])

    def _build_tree(self, data: np.ndarray, targets: np.ndarray, depth: int = 0):
        """
        Recursively builds the decision tree

        Args:
            - data (np.ndarray): The input data matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input target matrix, must have shape [n_samples, ]
            - depth (int): Current depth of tree

        Returns:
            - Node: Recursively this would provide the _root node!
        """
        n_samples, n_classes = targets.shape[0], len(np.unique(targets))

        # Stopping conditions
        if depth == self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            return Node(value=np.bincount(targets).argmax())

        # Select random subset of predict_data if n_features is set
        selected_feature_idxs = np.random.choice(data.shape[1], self.n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(data, targets, selected_feature_idxs)

        # If no valid split, return a leaf node
        if best_feature is None:
            return Node(value=np.bincount(targets).argmax())

        # Split the data
        left_idxs, right_idxs = self._split(data[:, best_feature], best_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=np.bincount(targets).argmax())

        # Build child nodes
        left_subtree = self._build_tree(data[left_idxs], targets[left_idxs], depth + 1)
        right_subtree = self._build_tree(data[right_idxs], targets[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    @staticmethod
    def _find_best_split(data: np.ndarray, targets: np.ndarray, selected_feature_idxs: np.ndarray) -> tuple[int, int | float]:
        """
        Finds the best feature and threshold to split on

        Args:
            - data (np.ndarray): Input data matrix of shape [n_samples, n_features]
            - targets (np.ndarray): Target vector of shape [n_samples, ]
            - selected_feature_idxs (np.ndarray): Indices of predict_data to consider for the split

        Returns:
            - tuple: Best feature index and best threshold for splitting, or (None, None) if no valid split is found
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in selected_feature_idxs:
            feature_column = data[:, feature_idx]
            thresholds = np.unique(feature_column)

            for threshold in thresholds:
                gain = information_gain(feature_column, targets, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_threshold = feature_idx, threshold

        return split_idx, split_threshold

    @staticmethod
    def _split(feature_column: np.ndarray, threshold: int | float) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into left and right subsets based on a threshold

        Args:
            - feature_column (np.ndarray): A single feature column of shape [n_samples, ]
            - threshold (int or float): The threshold value for splitting

        Returns:
            - tuple[np.ndarray, np.ndarray]: Indices for left and right splits
        """
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        return left_idxs, right_idxs

    def _traverse_tree(self, data_sample: np.ndarray, node: Node) -> int:
        """
        Traverses the tree to make a prediction for a single sample

        Args:
            - data_sample (np.ndarray): A single data sample, shape [n_features, ]
            - node (Node): The current node of the tree

        Returns:
            - int: Predicted class label
        """
        if node.value is not None:  # Check if node is leaf node
            return node.value
        if data_sample[node.feature] <= node.threshold:
            return self._traverse_tree(data_sample, node.left)
        else:
            return self._traverse_tree(data_sample, node.right)


if __name__ == '__main__':
    print('Testing Decision Tree algorithm')
    from examples import decision_tree
    decision_tree(visualize=True)
