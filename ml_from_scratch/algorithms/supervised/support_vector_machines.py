import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_predict_regression


class SupportVectorMachines:
    """
    A simple implementation of a Support Vector Machine (SVM) for binary classification
    using a linear kernel and hinge loss.
    """

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000) -> None:
        """
        Initializes the SVM model.

        Args:
            - learning_rate (float): The learning rate for gradient descent optimization. Defaults to 0.001.
            - lambda_param (float): Regularization parameter (controls the margin width). Defaults to 0.01.
            - n_iters (int): Number of iterations for training. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the SVM model to the training data using Stochastic Gradient Descent (SGD).

        Args:
            - features (ArrayLike): Training feature matrix of shape [n_samples, n_features].
            - targets (ArrayLike): Training labels of shape [n_samples, ].
                             Labels must be binary and are expected to be {0, 1} or {-1, 1}.
        """
        # validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)

        # Ensure labels are in {-1, 1}
        targets_ = np.where(targets <= 0, -1, 1)
        self._gradient_descent(features, targets_)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts class labels for the given input data.

        Args:
            - features (ArrayLike): Feature matrix of shape [n_samples, n_features].

        Returns:
            - np.ndarray: Predicted class labels of shape [n_samples, ].
                          Output labels are in {-1, 1}.
        """
        # validate inputs and convert to numpy arrays
        features = validate_predict_regression(features, self.weights, self.bias)
        approx = np.dot(features, self.weights) - self.bias
        return np.sign(approx)

    def _gradient_descent(self, features: np.ndarray, targets: np.ndarray):
        """
        Perform gradient descent on the training data

        Args:
            - features (np.ndarray): The input feature matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input predict_features matrix, must have shape [n_samples, ] or shape [n_samples, num_classes]
        """
        # Initialize weights and bias
        self.weights = np.zeros(features.shape[1])
        self.bias = 0

        # Gradient Descent for SVM optimization
        for _ in range(self.n_iters):
            for idx, feature_vector in enumerate(features):
                condition = targets[idx] * (np.dot(feature_vector, self.weights) - self.bias) >= 1
                if condition:
                    # Correctly classified, only regularization term
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Misclassified, update weights and bias
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(feature_vector, targets[idx]))
                    self.bias -= self.learning_rate * targets[idx]

