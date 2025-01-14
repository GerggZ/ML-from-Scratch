import numpy as np


def compute_pairwise_distances(vectors1: np.ndarray, vectors2: np.ndarray, metric: str = "euclidean)") -> np.ndarray:
    """
    Compute pairwise distances between two sets of vectors using NumPy.

    Args:
        vectors1 (np.ndarray): Array of shape [n_samples_1, n_features]
        vectors2 (np.ndarray): Array of shape [n_samples_2, n_features]
        metric (str): Distance metric to use. Supports: 'euclidean', 'manhattan', 'cosine'

    Returns:
        np.ndarray: Pairwise distance matrix of shape [n_samples_1, n_samples_2]
    """
    if metric == "euclidean":
        # Compute Euclidean distance
        return np.sqrt(np.sum((vectors1[:, None, :] - vectors2[None, :, :]) ** 2, axis=2))
    elif metric == "manhattan":
        # Compute Manhattan distance
        return np.sum(np.abs(vectors1[:, None, :] - vectors2[None, :, :]), axis=2)
    elif metric == "cosine":
        # Compute Cosine distance
        vectors1_norm = np.linalg.norm(vectors1, axis=1, keepdims=True)
        vectors2_norm = np.linalg.norm(vectors2, axis=1, keepdims=True)

        # Handle zero vectors to avoid division by zero
        if np.any(vectors1_norm == 0) or np.any(vectors2_norm == 0):
            raise ValueError("Cosine distance is undefined for zero vectors.")

        vectors1_normalized = vectors1 / vectors1_norm
        vectors2_normalized = vectors2 / vectors2_norm
        return 1 - np.dot(vectors1_normalized, vectors2_normalized.T)

    else:
        raise ValueError(f"Unsupported metric '{metric}'. Supported metrics: 'euclidean', 'manhattan', 'cosine'.")
