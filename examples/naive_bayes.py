from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import NaiveBayes
from ml_from_scratch.utils.metrics import accuracy_score
from data_bases.get_database import get_sklearn_data_split


def naive_bayes_example(
        X_train, X_test, y_train, y_test,
        visualize: bool = False
):
    """
    Example of how to use the Naive Bayes algorithm with the sklearn iris dataset
    Optionally visualizes the output using matplotlib
    """
    pca = PCA(n_components=10)  # Provides better results when components are less correlated
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"\tNaive Bayes accuracy: {acc:.4f}")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_classification_supervised

        pca = PCA(n_components=2)  # Projecting data to 2D for visualization
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)

        plot_classification_supervised(
            X_train_2d, X_test_2d, y_train, y_test, predictions,
            title="Random Forest Iris Classification\n(PCA-reduced)",
            xlabel="", ylabel="", labels=["Setosa", "Versicolor", "Virginica"],
            supplimental_text=f"Accuracy: {acc:.3f}"
        )


if __name__ == '__main__':
    print('Testing Naive Bayes algorithm')
    train_test_data = get_sklearn_data_split("digits", test_size=0.3, random_state=42)
    naive_bayes_example(*train_test_data, visualize=True)
