import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_predict_regression
from ml_from_scratch.utils.activation_functions import sigmoid, softmax


class LogisticRegression:
    """
    A Simple Logistic Regression model made using NumPy
    """

    def __init__(self, learning_rate: float, num_iterations: int, activation_function: callable = sigmoid, num_classes: int | None = None):
        """
        Initializes necessary parameters

        Args:
            - learning_rate (float): The step size for gradient descent
            - num_iterations (int): The number of iterations (i.e., epochs) to optimize
            - activation_function (callable): Activation function to use when classifying class_predictions (e.g., `sigmoid` or `softmax`)
            - num_classes Optional(int): Number of classes (required if _activation_function is `softmax`).
        """
        if activation_function not in [sigmoid, softmax]:
            raise ValueError("Invalid _activation_function. Choose either `sigmoid` or `softmax`.")
        if activation_function == softmax and num_classes is None:
            raise ValueError("num_classes must be specified when using softmax activation.")

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self._activation_function = activation_function
        self.num_classes = num_classes

        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | float | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Linear Regression model to the provided training data using gradient descent

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - predict_features (ArrayLike): The input predict_features matrix, must have shape [n_samples, ] or shape [n_samples, num_classes]
        """
        # validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)

        # One-hot encode predict_features if using softmax
        if self._activation_function == softmax:
            targets = np.eye(self.num_classes)[targets]

        # perform gradient descent
        self._gradient_descent(features, targets)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data (predict_features)

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, num_classes]
        """
        # validate inputs and convert to numpy arrays
        features = validate_predict_regression(features, self.weights, self.bias)

        # compute class_predictions
        linear_predictions = np.dot(features, self.weights) + self.bias
        target_probabilities = self._activation_function(linear_predictions)

        if self._activation_function == sigmoid:
            class_predictions = (target_probabilities >= 0.5).astype(int)
        else:  # self._activation_function == softmax:
            class_predictions = np.argmax(target_probabilities, axis=1)

        return class_predictions

    def _gradient_descent(self, features, targets):
        """
        Perform gradient descent on the training data

        Args:
            - features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input predict_features matrix, must have shape [n_samples, ] or shape [n_samples, num_classes]
        """
        num_samples, num_features = features.shape
        self._initialize_weights_and_bias(num_features)

        for _ in range(self.num_iterations):
            # calculate class_predictions
            linear_predictions = np.dot(features, self.weights) + self.bias
            class_predictions = self._activation_function(linear_predictions)

            # determine delta variables for weight (dw) and bias (db)
            dw = (1 / num_samples) * np.dot(features.T, (class_predictions - targets))
            db = (1 / num_samples) * np.sum(class_predictions - targets, axis=0)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _initialize_weights_and_bias(self, num_features):
        """
        Initializes weights and bias to zero
        """
        if self._activation_function == sigmoid:
            self.weights = np.zeros(num_features)
            self.bias = 0.0
        else:  # self._activation_function == softmax:
            self.weights = np.zeros((num_features, self.num_classes))
            self.bias = np.zeros(self.num_classes)


if __name__ == '__main__':
    print('Testing Logistic Regression algorithm')
    from examples import logistic_regression, logistic_regression_multiclass

    logistic_regression(visualize=True)
    logistic_regression_multiclass(visualize=True)
