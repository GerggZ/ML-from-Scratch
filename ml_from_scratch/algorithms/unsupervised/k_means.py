import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.utils.metrics import compute_distances_to_point
from ml_from_scratch.utils.validate_inputs import validate_pos_int_param

class KMeans:
    """
    Implementation of the K-Means clustering algorithm.
    """

    def __init__(self, n_clusters: int = 5, max_iterations: int = 100, distance_metric: str = "euclidean"):
        """
        Initializes the KMeans model.

        Args:
            - n_clusters (int): Number of clusters (K). Defaults to 5.
            - max_iterations (int): Maximum number of iterations for convergence. Defaults to 100.
            - distance_metric (str): Distance metric to use. Defaults to 'euclidean'.
        """
        self.n_clusters = validate_pos_int_param(n_clusters)
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric

        self.data = None

    def fit_predict(self, data: ArrayLike) -> np.ndarray:
        """
        Fits the KMeans model to the data and predicts cluster assignments.

        Args:
            - data (ArrayLike): The input data matrix, must have shape [n_samples, n_features]

        Returns:
            - np.ndarray: Cluster labels for each sample.
        """
        self.data = np.array(data)

        cluster_labels = self._cluster_data()
        return cluster_labels

    def _cluster_data(self) -> np.ndarray:
        """
        Core clustering method.
        Initializes and iterates to update centroids

        Returns:
            - np.ndarray: cluster labels for each sample
        """
        n_samples, n_features = self.data.shape

        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        new_centroids = self.data[random_indices]

        # Iterate to adjust centroid locations
        for _ in range(self.max_iterations):
            cluster_labels = self._assign_clusters(new_centroids)

            # Calculate new centroids from the clusters
            old_centroids = new_centroids
            new_centroids = self._calculate_centroids(cluster_labels, n_features)

            # Check for convergence
            cluster_movement = np.allclose(old_centroids, new_centroids, atol=1e-6)
            if cluster_movement:
                break

        return cluster_labels

    def _assign_clusters(self, centroids) -> np.ndarray:
        """
        Assigns each data point to the closest centroid.

        Returns:
            - np.ndarray: Cluster labels for each sample.
        """
        distances = np.array([
            compute_distances_to_point(self.data, centroid, metric="euclidean")
            for centroid in centroids
        ])
        return np.argmin(distances, axis=0)

    def _calculate_centroids(self, cluster_labels: np.ndarray, n_features: int) -> np.ndarray:
        """
        Calculates new centroids as the mean of samples in each cluster.

        Args:
            - cluster_labels (np.ndarray): Cluster labels for each sample
            - n_features (int): Number of features for each sample
        Returns:
            - np.ndarray: New centroids of shape [n_clusters, n_features]
        """
        centroids = np.array([
            self.data[cluster_labels == cluster_idx].mean(axis=0)
            if np.any(cluster_labels == cluster_idx) else np.zeros(n_features)
            for cluster_idx in range(self.n_clusters)
        ])
        return centroids
