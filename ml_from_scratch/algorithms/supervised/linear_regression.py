import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_predict_regression


class LinearRegression:
    """
    A Simple Linear Regression model made using NumPy
    """
    def __init__(self, learning_rate: float, num_iterations: int) -> None:
        """
        Initializes necessary parameters

        Args:
            - learning_rate (float): The step size for gradient descent
            - num_iterations (int): The number of iterations (i.e., epochs) to optimize
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Linear Regression model to the provided training data using gradient descent

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - predict_features (ArrayLike): The input predict_features matrix, must have shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)

        # Perform gradient descent
        self._gradient_descent(features, targets)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data (predict_features)

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features = validate_predict_regression(features, self.weights, self.bias)

        # Compute class_predictions
        predicted_targets = np.dot(features, self.weights) + self.bias

        return predicted_targets

    def _gradient_descent(self, features, targets):
        """
        Perform gradient descent on the training data

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - predict_features (ArrayLike): The input predict_features matrix, must have shape [n_samples, ]
        """
        num_samples, num_features = features.shape
        self._initialize_weights_and_bias(num_features)

        for _ in range(self.num_iterations):
            # calculate class_predictions
            target_predictions = np.dot(features, self.weights) + self.bias

            # determine delta variables for weight (dw) and bias (db)
            dw = (1/num_samples) * np.dot(features.T, (target_predictions - targets))
            db = (1/num_samples) * np.sum(target_predictions - targets)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _initialize_weights_and_bias(self, num_features: int):
        self.weights = np.zeros(num_features)
        self.bias = 0.0


if __name__ == '__main__':
    print('Testing Linear Regression algorithm')
    from examples import linear_regression
    linear_regression(visualize=True)
