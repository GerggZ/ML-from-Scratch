import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_regression
from ml_from_scratch.utils.activation_functions import sigmoid, softmax


class LogisticRegression:
    """
    A Simple Logistic Regression model made using NumPy
    """

    def __init__(self, learning_rate: float, n_iterations: int, activation_function: callable = sigmoid, n_classes: int | None = None):
        """
        Initializes necessary parameters

        Args:
            - learning_rate (float): The step size for gradient descent
            - n_iterations (int): The number of iterations (i.e., epochs) to optimize
            - activation_function (callable): Activation function to use when classifying class_predictions (e.g., `sigmoid` or `softmax`)
        Optional Args:
            - n_classes (int): Number of classes (required if activation_function is `softmax`).
        """
        if activation_function not in [sigmoid, softmax]:
            raise ValueError("Invalid activation_function. Choose either `sigmoid` or `softmax`.")
        if activation_function == softmax and n_classes is None:
            raise ValueError("n_classes must be specified when using softmax activation.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_function = activation_function
        self.n_classes = n_classes

        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | float | None = None

        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Logistic Regression algorithm using the provided training data and targets

        Args
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ] (binary)
                                                            must have shape [n_samples, n_classes] (multiclass)
        """
        data, targets = validate_fit(data, targets)
        self.training_data_shape = data.shape

        # One-hot encode predict_data if using softmax
        if self.activation_function == softmax:
            targets = np.eye(self.n_classes)[targets]

        self._gradient_descent(data, targets)

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ] (binary)
                                                   shape [m_samples, n_classes] (multiclass)
        """
        data = validate_predict_regression(data, self._weights, self._bias)

        # compute class_predictions
        linear_predictions = np.dot(data, self._weights) + self._bias
        target_probabilities = self.activation_function(linear_predictions)

        if self.activation_function == sigmoid:
            class_predictions = (target_probabilities >= 0.5).astype(int)
        else:  # self.activation_function == softmax:
            class_predictions = np.argmax(target_probabilities, axis=1)

        return class_predictions

    def _gradient_descent(self, data: np.ndarray, targets: np.ndarray):
        """
        Perform gradient descent on the training data

        Args:
            - data (np.ndarray): The input data matrix, must have shape [n_samples, n_features]
            - targets (np.ndarray): The input target matrix, must have shape [n_samples, ] (binary)
                                                             must have shape [n_samples, n_classes] (multiclass)
        """
        num_samples, num_features = data.shape
        self._initialize_weights_and_bias(num_features)

        for _ in range(self.n_iterations):
            # calculate class_predictions
            linear_predictions = np.dot(data, self._weights) + self._bias
            class_predictions = self.activation_function(linear_predictions)

            # determine delta variables for weight (dw) and _bias (db)
            dw = (1 / num_samples) * np.dot(data.T, (class_predictions - targets))
            db = (1 / num_samples) * np.sum(class_predictions - targets, axis=0)

            # update _weights and _bias
            self._weights -= self.learning_rate * dw
            self._bias -= self.learning_rate * db

    def _initialize_weights_and_bias(self, n_features):
        """
        Initializes weights and bias to zero

        Args:
            - n_features (int): Number of features in the training data
        """
        if self.activation_function == sigmoid:
            self._weights = np.zeros(n_features)
            self._bias = 0.0
        else:  # self.activation_function == softmax:
            self._weights = np.zeros((n_features, self.n_classes))
            self._bias = np.zeros(self.n_classes)


if __name__ == '__main__':
    print('Testing Logistic Regression algorithm')
    from examples import logistic_regression, logistic_regression_multiclass

    logistic_regression(visualize=True)
    logistic_regression_multiclass(visualize=True)
