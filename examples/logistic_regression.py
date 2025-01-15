import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

from ml_from_scratch.algorithms import LogisticRegression
from ml_from_scratch.utils.activation_functions import sigmoid, softmax
from ml_from_scratch.utils.metrics import accuracy_score


def logistic_regression(visualize: bool = False):
    """
    Tests binary logistic regression using the sklearn breast_cancer dataset.
    Optionally visualizes the decision boundary with true labels.
    """
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the predict_features
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
    acc = accuracy_score(y_test, predictions)
    print(f"Binary Logistic Regression Accuracy: {acc:.4f}")

    # Visualization
    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_logistic_regression

        # Reduce data to 2D (for visualization)
        pca = PCA(n_components=2)
        pca.fit(X_train)
        X_test_2d = pca.transform(X_test)

        # Generate a meshgrid for the decision boundaries
        grid_size = 50
        x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
        y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

        # Create grid points in 2D space, transform it into the dimensionality of the data, and then classify it
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_points_original_d = pca.inverse_transform(grid_points_2d)
        grid_predictions = model.predict(grid_points_original_d).reshape(xx.shape)

        plot_logistic_regression(
            X_test, y_test, predictions,
            xx, yy, grid_predictions,
            title="Binary Logistic Regression with Decision Boundary (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"ACC: {acc:.4f}"
        )


def logistic_regression_multiclass(visualize: bool = False):
    """
    Tests multiclass logistic regression using the sklearn digits dataset.
    Optionally visualizes the decision boundary with true labels.
    """
    # Load the dataset
    data = load_digits()
    X, y = data.data, data.target
    num_classes = len(np.unique(y))

    # Standardize the predict_features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression Model
    reg = LogisticRegression(
        learning_rate=0.01,
        num_iterations=1000,
        activation_function=softmax,
        num_classes=num_classes,
    )
    reg.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = reg.predict(X_test)

    # Evaluate accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Multiclass Logistic Regression Accuracy: {acc:.4f}")

    # Visualization
    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_logistic_regression

        if X_train.shape[1] > 2:
            # Reduce data to 2D (for visualization)
            pca = PCA(n_components=2)
            pca.fit(X_train)
            X_test = pca.transform(X_test)

            # Generate a meshgrid for the decision boundaries
            grid_size = 50
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

            # Create grid points in 2D space, transform it into the dimensionality of the data, and hten classify it
            grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_points_original_d = pca.inverse_transform(grid_points_2d)
            grid_predictions = reg.predict(grid_points_original_d).reshape(xx.shape)
        else:
            # Generate a meshgrid for the decision boundaries
            grid_size = 50
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

            # Create grid points in 2D space, transform it into the dimensionality of the data, and hten classify it
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_predictions = reg.predict(grid_points).reshape(xx.shape)

        plot_logistic_regression(
            X_test, y_test, predictions,
            xx, yy, grid_predictions,
            title="Multiclass Logistic Regression with Decision Boundaries (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"ACC: {acc:.4f}"
        )


if __name__ == '__main__':
    print('Testing Logistic Regression algorithm')
    logistic_regression(visualize=True)
    logistic_regression_multiclass(visualize=True)
