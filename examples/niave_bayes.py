from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import NaiveBayes
from ml_from_scratch.utils.metrics import accuracy_score


def naive_bayes(visualize:bool = False):

    # Load and split the dataset
    data = load_iris()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Naive Bayes Accuracy: {acc:.4f}")

    # Visualization
    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_classification_supervised

        # Reduce data to 2D (for visualization)
        pca = PCA(n_components=2)
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
    naive_bayes(visualize=True)
