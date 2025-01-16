import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_regression


class LinearRegression:
    """
    A Simple Linear Regression model made using NumPy
    """
    def __init__(self, learning_rate: float, n_iterations: int) -> None:
        """
        Initializes necessary parameters

        Args:
            - learning_rate (float): The step size for gradient descent
            - n_iterations (int): The number of iterations (i.e., epochs) to optimize
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self._weights: np.ndarray | None = None
        self._bias: float | None = None

        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Linear Regression algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ]
        """
        data, targets = validate_fit(data, targets)
        self.training_data_shape = data.shape

        self._gradient_descent(data, targets)

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ]
        """
        data = validate_predict_regression(data, self._weights, self._bias)
        predicted_targets = np.dot(data, self._weights) + self._bias

        return predicted_targets

    def _gradient_descent(self, data: np.ndarray, targets: np.ndarray) -> None:
        """
        Perform gradient descent on the training data

        Args:
            - data (np.ndarray): The input data matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input target matrix, must have shape [n_samples, ]
        """
        n_samples, n_features = data.shape
        self._initialize_weights_and_bias(n_features)

        for _ in range(self.n_iterations):
            target_predictions = np.dot(data, self._weights) + self._bias

            # determine delta variables for weight (dw) and _bias (db)
            dw = (1/n_samples) * np.dot(data.T, (target_predictions - targets))
            db = (1/n_samples) * np.sum(target_predictions - targets)

            # update _weights and _bias
            self._weights -= self.learning_rate * dw
            self._bias -= self.learning_rate * db

    def _initialize_weights_and_bias(self, n_features: int) -> None:
        """
        Initializes weights and bias to zero

        Args:
            - n_features (int): Number of features in the training data
        """
        self._weights = np.zeros(n_features)
        self._bias = 0.0


if __name__ == '__main__':
    print('Testing Linear Regression algorithm')
    from examples import linear_regression_example
    linear_regression_example(visualize=True)
