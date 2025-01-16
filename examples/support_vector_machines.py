import numpy as np
from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import SupportVectorMachines
from ml_from_scratch.utils.metrics import accuracy_score
from data_bases.get_database import get_sklearn_data_split


def support_vector_machines_binary_example(
        X_train, X_test, y_train, y_test,
        visualize: bool = False
):
    """
    Example of how to use the Support Vector Machines (SVM) algorithm with the sklearn breast cancer dataset
    Optionally visualizes the output using matplotlib
    """
    model = SupportVectorMachines(learning_rate=0.01, n_iterations=1000, lambda_param=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.where(predictions == -1, 0, 1)

    acc = accuracy_score(y_test, predictions)
    print(f"\tSupport Vector Machines accuracy: {acc:.4f}")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_logistic_regression

        pca = PCA(n_components=2)  # Projecting data to 2D for visualization
        pca.fit(X_train)
        X_test_2d = pca.transform(X_test)

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
            title="Support Vector Machines with Decision Boundary (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"ACC: {acc:.4f}"
        )


if __name__ == '__main__':
    print('Testing Support Vector Machines algorithm')
    train_test_data = get_sklearn_data_split("breast cancer", test_size=0.2, random_state=42)
    support_vector_machines_binary_example(*train_test_data, visualize=True)
