from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import KNearestNeighbors
from ml_from_scratch.utils.metrics import accuracy_score
from data_bases.get_database import get_sklearn_data_split


def k_nearest_neighbors_example(
        X_train, X_test, y_train, y_test,
        visualize: bool = False
):
    """
    Example of how to use the K Nearest Neighbors algorithm with the sklearn iris dataset
    Optionally visualizes the output using matplotlib
    """
    cls = KNearestNeighbors(num_neighbors=5)
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"\tK Nearest Neighbors accuracy {acc:.4f}")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_classification_supervised

        pca = PCA(n_components=2)  # Projecting data to 2D for visualization
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)

        plot_classification_supervised(
            X_train_2d, X_test_2d, y_train, y_test, predictions,
            title="k-Nearest Neighbors Visualization (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"Accuracy: {acc:.3f}"
        )


if __name__ == '__main__':
    print('Testing K Nearest Neighbors algorithm')
    train_test_data = get_sklearn_data_split("iris", test_size=0.2, random_state=42)
    k_nearest_neighbors_example(*train_test_data, visualize=True)
