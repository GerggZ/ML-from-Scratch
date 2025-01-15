import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_fit, validate_predict_classifier


class NaiveBayes:
    """
    A Naive Bayes classifier for Gaussian-distributed data
    Assumes data are continuous and follow a normal (Gaussian) distribution
    """

    def __init__(self):
        """
        Initializes the NaiveBayes classifier.

        Attributes:
            - _classes (np.ndarray): Array of unique class labels.
            - _mean (np.ndarray): Array of shape [n_classes, n_features], storing the mean of each feature for each class.
            - _variance (np.ndarray): Array of shape [n_classes, n_features], storing the variance of each feature for each class.
            - _priors (np.ndarray): Array of shape [n_classes], storing the prior probabilities for each class.
        """
        self._classes = None
        self._mean = None
        self._variances = None
        self._priors = None

        self.training_data_shape = None

    def fit(self, data: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Decision Tree model to the provided training data using gradient descent

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]
            - targets (ArrayLike): The input target matrix, must have shape [n_samples, ]
        """
        data, targets = validate_fit(data, targets)
        self.training_data_shape = data.shape
        self._classes = np.unique(targets)
        n_samples, n_features = self.training_data_shape

        self._initialize_mean_variance_priors()

        # Calculate mean, variance, and prior for each class
        for idx, cls in enumerate(self._classes):
            feature_class = data[targets == cls]
            self._mean[idx, :] = feature_class.mean(axis=0)
            self._variance[idx, :] = feature_class.var(axis=0)
            self._priors[idx] = feature_class.shape[0] / float(n_samples)

    def predict(self, data: ArrayLike) -> np.ndarray:
        """
        Fits the Naive Bayes algorithm using the provided training data and targets

        Args:
            - data (ArrayLike): The input data matrix, must have shape [m_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [m_samples, ]
        """
        data = validate_predict_classifier(data, self.training_data_shape)
        return np.array([self._predict(data_sample) for data_sample in data])

    def _initialize_mean_variance_priors(self) -> None:
        """
        Initializes the mean, variance, and prior matrices to zero
        """
        n_samples, n_features = self.training_data_shape
        n_classes = len(self._classes)

        # Initialize the mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

    def _predict(self, data_sample: np.ndarray) -> int:
        """
        Predicts target values for a single data sample

        Args:
            - data_sample (np.ndarray): A single data sample, shape [n_features, ]

        Returns:
            - int: Predicted class label
        """
        posteriors = []

        # Calculate the posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, data_sample)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx: int, data_sample: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function (PDF) of the Gaussian distribution
        for a given feature vector and class.

        Args:
            - class_idx (int): Index of the class.
            - data_sample (np.ndarray): A single data sample, shape [n_features, ]

        Returns:
            - np.ndarray: PDF values for the data sample, shape [n_features, ]
        """
        mean = self._mean[class_idx]
        var = self._variance[class_idx]

        numerator = np.exp(-((data_sample - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
