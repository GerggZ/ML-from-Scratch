import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

from ml_from_scratch.algorithms import LogisticRegression
from ml_from_scratch.utils.activation_functions import sigmoid, softmax


def test_binary_logistic_regression(visualize: bool = False):
    """
    Tests binary logistic regression using the sklearn breast_cancer dataset.
    Optionally visualizes the decision boundary with true labels.
    """
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reduce to 2D for visualization (if needed)
    if visualize:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000, activation_function=sigmoid)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Binary Logistic Regression Accuracy: {accuracy:.4f}")

    # Visualization
    if visualize:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))

        # Plot the decision boundary
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = model.predict(grid_points).reshape(xx.shape)

        plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap="coolwarm")

        # Plot the true labels with more meaningful names
        labels = ["Breast Cancer Negative", "Breast Cancer Positive"]
        scatter = plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c=y_test,
            cmap="coolwarm",
            edgecolor="k",
            label="True Labels"
        )

        # Update legend
        legend1 = plt.legend(
            handles=scatter.legend_elements()[0],
            labels=labels,
            title="True Labels",
            loc="upper right"
        )
        plt.gca().add_artist(legend1)

        plt.title("Binary Logistic Regression with Decision Boundary (PCA-reduced)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()


def test_multiclass_logistic_regression(visualize: bool = False):
    """
    Tests multiclass logistic regression using the sklearn digits dataset.
    Optionally visualizes the decision boundary with true labels.
    """
    # Load the dataset
    data = load_digits()
    X, y = data.data, data.target
    num_classes = len(np.unique(y))

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reduce to 2D for visualization (if needed)
    if visualize:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression(
        learning_rate=0.01,
        num_iterations=1000,
        activation_function=softmax,
        num_classes=num_classes,
    )
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Multiclass Logistic Regression Accuracy: {accuracy:.4f}")

    # Visualization
    if visualize:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))

        # Plot the decision boundary
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = model.predict(grid_points).reshape(xx.shape)

        plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap="tab10")

        # Plot the true labels
        scatter = plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c=y_test,
            cmap="tab10",
            edgecolor="k",
            label="True Labels"
        )

        plt.title("Multiclass Logistic Regression with Decision Boundaries (PCA-reduced)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
        plt.show()

