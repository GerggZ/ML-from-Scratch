from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import KNearestNeighbors
from ml_from_scratch.utils.metrics import accuracy_score


def k_nearest_neighbors(visualize: bool = False):
    """
    Tests k Nearest Neighbors using the sklearn breast_cancer dataset.
    Optionally visualizes the decision boundary with true labels.
    """
    # Create synthetic regression data
    iris_dataset = load_iris()
    X, y = iris_dataset.data, iris_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    cls = KNearestNeighbors(num_neighbors=5)
    cls.fit(X_train, y_train)

    # Make class predictions
    predictions = cls.predict(X_test)

    # Calculate the accuracy
    acc = accuracy_score(y_test, predictions)

    print(f"K Nearest Neighbors Model trained successfully")
    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_classification

        # Reduce data to 2D (for visualization)
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)

        plot_classification(
            X_train_2d, X_test_2d, y_train, y_test, predictions,
            title="k-Nearest Neighbors Visualization (PCA-reduced)", xlabel="", ylabel="",
            supplimental_text=f"Accuracy: {acc:.3f}"
        )


if __name__ == '__main__':
    print('Testing K Nearest Neighbors algorithm')
    k_nearest_neighbors(visualize=True)
