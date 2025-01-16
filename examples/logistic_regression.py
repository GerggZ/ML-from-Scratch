import numpy as np
from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import LogisticRegression
from ml_from_scratch.utils.activation_functions import sigmoid, softmax
from ml_from_scratch.utils.metrics import accuracy_score
from data_bases.get_database import get_sklearn_data_split


def logistic_regression_binary_example(
        X_train, X_test, y_train, y_test,
        visualize: bool = False
):
    """
    Example of how to use the Logistic Regression (binary) algorithm with the sklearn breast cancer dataset
    Optionally visualizes the output using matplotlib
    """
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000, activation_function=sigmoid)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"\tLogistic Regression (binary) accuracy: {acc:.4f}")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_logistic_regression

        pca = PCA(n_components=2)  # Projecting data to 2D for visualization
        pca.fit(X_train)
        X_test_2d = pca.transform(X_test)

        # Generate a meshgrid for the decision boundary visualization
        grid_size = 50
        x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
        y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

        grid_points_2d = np.c_[xx.ravel(), -yy.ravel()]
        grid_points_original_d = pca.inverse_transform(grid_points_2d)
        grid_predictions = model.predict(grid_points_original_d).reshape(xx.shape)

        plot_logistic_regression(
            X_test, y_test, predictions,
            xx, yy, grid_predictions,
            title="Binary Logistic Regression with Decision Boundary (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"ACC: {acc:.4f}"
        )


def logistic_regression_multiclass_example(
        X_train, X_test, y_train, y_test,
        visualize: bool = False
):
    """
    Example of how to use the Logistic Regression (multiclass) algorithm with the sklearn digits dataset
    Optionally visualizes the output using matplotlib
    """
    num_classes = len(np.union1d(y_train, y_test))
    reg = LogisticRegression(learning_rate=0.01, n_iterations=1000, activation_function=softmax, n_classes=num_classes)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"\tLogistic Regression (multiclass) accuracy: {acc:.4f}")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_logistic_regression

        if X_train.shape[1] > 2:
            pca = PCA(n_components=2)  # Projecting data to 2D for visualization
            pca.fit(X_train)
            X_test = pca.transform(X_test)

            # Generate a meshgrid for the decision boundary visualization
            grid_size = 50
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

            grid_points_2d = np.c_[xx.ravel(), -yy.ravel()]
            grid_points_original_d = pca.inverse_transform(grid_points_2d)
            grid_predictions = reg.predict(grid_points_original_d).reshape(xx.shape)
        else:
            # Generate a meshgrid for the decision boundary visualization
            grid_size = 50
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

            grid_points = np.c_[xx.ravel(), -yy.ravel()]
            grid_predictions = reg.predict(grid_points).reshape(xx.shape)

        plot_logistic_regression(
            X_test, y_test, predictions,
            xx, yy, grid_predictions,
            title="Multiclass Logistic Regression with Decision Boundaries (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"ACC: {acc:.4f}"
        )


if __name__ == '__main__':
    print('Testing Logistic Regression (binary) algorithm')
    train_test_data = get_sklearn_data_split("breast cancer", test_size=0.2, random_state=42)
    logistic_regression_binary_example(*train_test_data, visualize=True)

    print('Testing Logistic Regression (multiclass) algorithm')
    train_test_data = get_sklearn_data_split("digits", test_size=0.2, random_state=42)
    logistic_regression_multiclass_example(*train_test_data, visualize=True)

