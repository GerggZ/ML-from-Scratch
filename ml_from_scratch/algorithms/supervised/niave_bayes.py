import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.validate_inputs import validate_supervised_fit, validate_predict_classifier

class NaiveBayes:
    """
    A Naive Bayes classifier for Gaussian-distributed data
    Assumes features are continuous and follow a normal (Gaussian) distribution
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

    def fit(self, features: ArrayLike, targets: ArrayLike) -> None:
        """
        Fits the Linear Regression model to the provided training data using gradient descent

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]
            - predict_features (ArrayLike): The input predict_features matrix, must have shape [n_samples, ]
        """
        # Validate inputs and convert to numpy arrays
        features, targets = validate_supervised_fit(features, targets)
        self.training_features_shape = features.shape  # stored for validating prediction inputs
        n_samples, n_features = self.training_features_shape

        self._classes = np.unique(targets)
        n_classes = len(self._classes)

        # Initialize the mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and prior for each class
        for idx, cls in enumerate(self._classes):
            feature_class = features[targets == cls]
            self._mean[idx, :] = feature_class.mean(axis=0)
            self._variance[idx, :] = feature_class.var(axis=0)
            self._priors[idx] = feature_class.shape[0] / float(n_samples)

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predicts target values for the given input data (predict_features)

        Args:
            - predict_features (ArrayLike): The input feature matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Predicted target values, shape [n_samples, num_classes]
        """
        # validate inputs and convert to numpy arrays
        features = validate_predict_classifier(features, np.zeros(self.training_features_shape))
        return np.array([self._predict(feature_vector) for feature_vector in features])

    def _predict(self, feature_vector: np.ndarray) -> int:
        """
        Predicts the class label for a single sample

        Args:
            - feature_vector (np.ndarray): A single feature vector of shape [n_features, ]

        Returns:
            - int: Predicted class label
        """
        posteriors = []

        # Calculate the posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, feature_vector)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx: int, feature_vector: np.ndarray) -> np.ndarray:
        """
        Computes the probability density function (PDF) of the Gaussian distribution
        for a given feature vector and class.

        Args:
            - class_idx (int): Index of the class.
            - feature_vector (np.ndarray): A single feature vector of shape [n_features, ].

        Returns:
            - np.ndarray: PDF values for each feature, shape [n_features, ].
        """
        mean = self._mean[class_idx]
        var = self._variance[class_idx]

        numerator = np.exp(-((feature_vector - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
