# ml_from_scratch\utils\metrics\__init__.py

from ml_from_scratch.utils.metrics.regression_metrics import mean_squared_error, mean_absolute_error, r_squared_score
from ml_from_scratch.utils.metrics.classification_metrics import accuracy_score, precision_score, recall_score, f1_score
from ml_from_scratch.utils.metrics.distance_metrics import compute_pairwise_distances, compute_distances_to_point

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared_score",
    "compute_pairwise_distances",
    "compute_distances_to_point",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
]

