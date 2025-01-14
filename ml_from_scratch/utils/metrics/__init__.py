# ml_from_scratch\utils\metrics\__init__.py

from .regression_metrics import mean_squared_error, mean_absolute_error, r_squared_score
from .classification_metrics import accuracy_score, precision_score, recall_score, f1_score
from .distance_metrics import compute_pairwise_distances

# Export all metrics in this namespace
__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared_score",
    "compute_pairwise_distances",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
]

# Clean up internal modules to prevent them from appearing in autocomplete
del _classification_metrics
del _distance_metrics
del _regression_metrics
