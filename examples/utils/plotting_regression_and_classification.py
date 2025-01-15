import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def plot_classification_supervised(
        X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, predictions: np.ndarray,
        title: str = "", xlabel: str = "", ylabel: str = "", labels: list = [], supplimental_text: str = ""
) -> None:
    # Compute global vmin and vmax for consistent color scaling
    unique_labels = np.unique(np.concatenate((y_test, predictions)))
    vmin, vmax = unique_labels.min(), unique_labels.max()

    # select color scheme based on number of classes:
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20' if len(unique_labels) <= 20 else 'viridis')

    plt.figure(figsize=(10, 8))

    # Plot training data
    plt.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train,
        cmap=cmap, vmin=vmin, vmax=vmax, s=50, marker="^", alpha=0.6, edgecolor="gray"
    )

    # Plot correctly classified test data
    correctly_classified = predictions == y_test
    plt.scatter(
        X_test[correctly_classified, 0], X_test[correctly_classified, 1], c=y_test[correctly_classified],
        cmap=cmap, vmin=vmin, vmax=vmax, s=70, marker="o", edgecolor="black"
    )

    # Plot misclassified test data
    misclassified = ~correctly_classified
    plt.scatter(
        X_test[misclassified, 0], X_test[misclassified, 1], c=y_test[misclassified],
        cmap=cmap, vmin=vmin, vmax=vmax, s=70, marker="x",
    )

    # Titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add supplimental text to the plot (e.g., accuracy, f1-score, etc.
    if len(supplimental_text):
        plt.text(
            0.05, 0.95,
            s=supplimental_text,
            fontsize=12,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
        )

    # Create custom legend
    legend_handles = [
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=10, label="Training Data"),
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=10,
                      label="Correctly Classified Test Data"),
        mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=10,
                      label="Misclassified Test Data"),
    ]
    # Add class-specific legend items if labels are provided
    if labels and len(labels) == len(unique_labels):
        for class_idx, class_label in enumerate(unique_labels):
            legend_handles.append(
                mlines.Line2D([], [], color=cmap((class_label - vmin) / (vmax - vmin)),
                              marker="s", linestyle="None", markersize=10, label=labels[class_idx])
            )
    plt.legend(handles=legend_handles, loc="best", fontsize=10, frameon=True)

    # Show the plot
    plt.show()


def plot_classification_unsupervised(
        X_data: np.ndarray, y_targets: np.ndarray, predictions: np.ndarray,
        ltitle: str = "", rtitle: str = "", supplimental_text: str = "",
        xlabel: str = "", ylabel: str = "", labels: list = [],
) -> None:

    # Compute global vmin and vmax for consistent color scaling
    unique_labels = np.unique(np.concatenate((y_targets, predictions)))
    vmin, vmax = unique_labels.min(), unique_labels.max()

    # select color scheme based on number of classes:
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20' if len(unique_labels) <= 20 else 'viridis')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # KMeans Predicted Clusters
    axes[0].scatter(
        X_data[:, 0], X_data[:, 1], c=predictions,
        cmap=cmap, vmin=vmin, vmax=vmax, s=50, marker="o", alpha=0.7
    )
    axes[0].set_title(ltitle, fontsize=14)
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Ground Truth Clusters
    axes[1].scatter(
        X_data[:, 0], X_data[:, 1], c=y_targets,
        cmap=cmap, vmin=vmin, vmax=vmax, s=50, marker="o", alpha=0.7
    )
    axes[1].set_title(rtitle, fontsize=14)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.scatter(
        X_data[:, 0], X_data[:, 1], c=y_targets,
        cmap=cmap, vmin=vmin, vmax=vmax, s=50, marker="o", label="Correctly Classified", alpha=0.7
    )

    # Add supplimental text to the plot (e.g., accuracy, f1-score, etc.
    if len(supplimental_text):
        axes[1].text(
            0.05, 0.95,
            s=supplimental_text,
            fontsize=12,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
        )

    # Add class-specific legend items if labels are provided
    if labels and len(labels) == len(unique_labels):
        handles = [
            plt.Line2D([], [], color=cmap((label - vmin) / (vmax - vmin)), marker="o", linestyle="None", markersize=10, label=labels[idx])
            for idx, label in enumerate(unique_labels)
        ]
        axes[0].legend(handles=handles, loc="best", fontsize=10, title="KMeans Clusters")
        axes[1].legend(handles=handles, loc="best", fontsize=10, title="True Labels")

    plt.show()


def plot_linear_regression(
    X: np.ndarray, y: np.ndarray, predictions: np.ndarray,
    title: str = "", xlabel: str = "", ylabel: str = "",
    supplimental_text: str = ""
) -> None:
    """
    Plots the results of a linear regression model.

    Args:
        - features (np.ndarray): Input features (1D or 2D with a single column).
        - targets (np.ndarray): Ground truth target values.
        - predictions (np.ndarray): Predicted target values by the model.
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the targets-axis.
    """
    # Flatten features if it's 2D with a single column
    if X.ndim > 1 and X.shape[1] == 1:
        X = X.ravel()

    plt.figure(figsize=(10, 6))

    # Plot the actual data points
    plt.scatter(X, y, color="blue", alpha=0.7, label="True Data Points", edgecolor="k")

    # Plot the regression line
    plt.plot(X, predictions, color="red", linewidth=2, label="Regression Line")

    # Add residual lines (distances from data points to regression line)
    for xi, yi, pi in zip(X, y, predictions):
        plt.plot([xi, xi], [yi, pi], color="gray", linestyle="dotted", alpha=0.5)

    # Titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Add legend
    plt.legend(loc="best", fontsize=10)

    # Show grid
    plt.grid(alpha=0.3, linestyle="--")

    # Add supplimental text to the plot (e.g., accuracy, f1-score, etc.
    if len(supplimental_text):
        plt.text(
            0.05, 0.95,
            s=supplimental_text,
            fontsize=12,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
        )

    # Show the plot
    plt.show()


def plot_logistic_regression(
        X_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray,
        xx: np.ndarray, yy: np.ndarray, grid_predictions: np.ndarray,
        title: str = "", xlabel: str = "", ylabel: str = "", supplimental_text: str = ""
) -> None:
    # Compute global vmin and vmax for consistent color scaling
    unique_labels = np.unique(np.concatenate((y_test, predictions)))
    vmin, vmax = unique_labels.min(), unique_labels.max()

    # select color scheme based on number of classes:
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20' if len(unique_labels) <= 20 else 'viridis')

    plt.figure(figsize=(10, 8))

    # Plot the decision boundary
    levels = np.concatenate(([unique_labels[0] - 0.5], unique_labels + 0.5))
    plt.contourf(xx, yy, grid_predictions, levels=levels, alpha=0.3, cmap=cmap, vmin=vmin, vmax=vmax)

    # Correctly and incorrectly classified indices
    correctly_classified = predictions == y_test
    misclassified = ~correctly_classified

    # Plot the true labels with some transparency
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test,
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=50, marker="o", alpha=0.4, edgecolor="none",
        label="True Labels (Background)"
    )

    # Plot correctly classified test data
    plt.scatter(
        X_test[correctly_classified, 0], X_test[correctly_classified, 1], c=y_test[correctly_classified],
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=50, marker="o", edgecolor="black",
        label="Correctly Classified"
    )

    # Plot misclassified test data
    plt.scatter(
        X_test[misclassified, 0], X_test[misclassified, 1], c=predictions[misclassified],
        cmap=cmap, vmin=vmin, vmax=vmax, s=50, marker="x",
        label="Misclassified"
    )

    # Titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Add supplimental text to the plot (e.g., accuracy, f1-score, etc.
    if len(supplimental_text):
        plt.text(
            0.05, 0.95,
            s=supplimental_text,
            fontsize=12,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
        )

    # Create custom legend
    legend_handles = [
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=10, label="Correctly Classified"),
        mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=10, label="Misclassified")
    ]
    plt.legend(handles=legend_handles, loc="best", fontsize=10, frameon=True)

    # Show the plot
    plt.show()
