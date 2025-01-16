import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_regression


class SupportVectorMachines:
    """
    A simple implementation of a Support Vector Machine (SVM) for binary classification
    using a linear kernel and hinge loss.
    """

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iterations: int = 1000) -> None:
        """
        Initializes the SVM model.

        Args:
            - learning_rate (float): The learning rate for gradient descent optimization
            - lambda_param (float): Regularization parameter (controls the margin width)
            - n_iterations (int): Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self._weights = None
        self._bias = None

        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Support Vector Machines algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ]
                                   targets must be binary and are expected to be {0, 1} or {-1, 1}
        """
        data, targets = validate_fit(data, targets)
        self.training_data_shape = data.shape

        # Ensure labels are in {-1, 1}
        targets_ = np.where(targets == 0, -1, 1)
        self._gradient_descent(data, targets_)

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ]
                          Output targets are in {-1, 1}.
        """
        data = validate_predict_regression(data, self._weights, self._bias)
        approx = np.dot(data, self._weights) - self._bias
        return np.sign(approx)

    def _gradient_descent(self, data: np.ndarray, targets: np.ndarray):
        """
        Perform gradient descent on the training data

        Args:
            - data (np.ndarray): The input data matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input target matrix, must have shape [n_samples, ]
        """
        self._initialize_weights_and_bias(data.shape[1])

        # Gradient Descent for SVM optimization
        for _ in range(self.n_iterations):
            for idx, data_sample in enumerate(data):
                condition = targets[idx] * (np.dot(data_sample, self._weights) - self._bias) >= 1
                if condition:
                    # Correctly classified, only regularization term
                    self._weights -= self.learning_rate * (2 * self.lambda_param * self._weights)
                else:
                    # Misclassified, update _weights and _bias
                    self._weights -= self.learning_rate * (2 * self.lambda_param * self._weights - np.dot(data_sample, targets[idx]))
                    self._bias -= self.learning_rate * targets[idx]

    def _initialize_weights_and_bias(self, num_features: int) -> None:
        """
        Initializes weights and bias to zero

        Args:
            - n_features (int): Number of features in the training data
        """
        self._weights = np.zeros(num_features)
        self._bias = 0.0
