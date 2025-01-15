import numpy as np
from numpy.typing import ArrayLike
from ml_from_scratch.utils.nodes.tree_nodes import Node
from ml_from_scratch.utils.math import information_gain_gaussian
from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_classifier


class GaussianDecisionTree:
    """
    A Gaussian Decision Tree that uses probabilistic splits based on Gaussian distributions
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,  n_features: int | None = None):
        """
        Initializes the Gaussian Decision Tree.

        Args:
            - max_depth (int, optional): Maximum depth of the tree
                                         If None, trees grow until other stopping criteria are met
            - min_samples_split (int): Minimum samples required to split a node
            - n_features (int): A number of data to randomly consider at each split (None uses all features)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self._root = None
        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Gaussian Decision Tree algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input predict_data matrix, must have shape [n_samples, ]
        """
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

    def _build_tree(self, data: np.ndarray, targets: np.ndarray, depth: int = 0) -> Node:
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
        best_feature, best_mean, best_std = self._find_best_gaussian_split(data, targets, selected_feature_idxs)

        # If no valid split, return a leaf node
        if best_feature is None:
            return Node(value=np.bincount(targets).argmax())

        # Split data based on Gaussian likelihood
        left_idxs, right_idxs = self._gaussian_split(data[:, best_feature], best_mean, best_std)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=np.bincount(targets).argmax())

        # Build child nodes
        left_subtree = self._build_tree(data[left_idxs], targets[left_idxs], depth + 1)
        right_subtree = self._build_tree(data[right_idxs], targets[right_idxs], depth + 1)
        return Node(feature=best_feature, mean=best_mean, std=best_std, left=left_subtree, right=right_subtree)

    @staticmethod
    def _find_best_gaussian_split(data: np.ndarray, targets: np.ndarray, selected_feature_idxs: list[int]):
        """
        Finds the best Gaussian split based on information gain

        Args:
            - data (np.ndarray): Input data matrix of shape [n_samples, n_features]
            - targets (np.ndarray): Target vector of shape [n_samples, ]
            - selected_feature_idxs (np.ndarray): Indices of predict_data to consider for the split

        Returns:
            - tuple: Best feature index, mean, and standard deviation
        """
        best_gain = -1
        best_feature, best_mean, best_std = None, None, None

        for feature_idx in selected_feature_idxs:
            feature_values = data[:, feature_idx]
            gain, mean, std = information_gain_gaussian(feature_values, targets)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_mean = mean
                best_std = std

        return best_feature, best_mean, best_std

    def _gaussian_split(self, feature_column: np.ndarray, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into left and right subsets based on a threshold

        Args:
            - feature_column (np.ndarray): A single feature column of shape [n_samples, ]
            - mean (float): The mean of the Gaussian distribution for this feature
            - std (float): The standard deviation of the Gaussian distribution for this feature


        Returns:
            - tuple[np.ndarray, np.ndarray]: Indices for left and right splits
        """
        likelihoods = self._gaussian_likelihood(feature_column, mean, std)

        # Split indices based on likelihood threshold (e.g., > 0.5)
        left_idxs = np.where(likelihoods > 0.5)[0]
        right_idxs = np.where(likelihoods <= 0.5)[0]

        return left_idxs, right_idxs

    @staticmethod
    def _gaussian_likelihood(values: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Computes the Gaussian likelihood for a set of values

        Args:
            - values (np.ndarray): Feature values of shape [n_samples, ]
            - mean (float): Mean of the Gaussian distribution
            - std (float): Standard deviation of the Gaussian distribution

        Returns:
            - np.ndarray: Gaussian likelihoods of shape [n_samples, ]
        """
        return np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def _traverse_tree(self, data_sample: np.ndarray, node: Node) -> int:
        """
        Traverses the tree to make a prediction for a single sample

        Args:
            - data_sample (np.ndarray): A single data sample, shape [n_features, ]
            - node (Node): The current node of the tree

        Returns:
            - int: Predicted class label.
        """
        if node.value is not None:
            return node.value

        likelihood = self._gaussian_likelihood(data_sample[node.feature], node.mean, node.std)

        if likelihood > 0.5:
            return self._traverse_tree(data_sample, node.left)
        else:
            return self._traverse_tree(data_sample, node.right)


if __name__ == '__main__':
    print('Testing Gaussian Decision Tree algorithm')
    from examples import gaussian_decision_tree
    gaussian_decision_tree(visualize=True)
