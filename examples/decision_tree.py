from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import DecisionTree
from ml_from_scratch.utils.metrics import accuracy_score


def decision_tree(visualize=False):
    """
    Tests decisiion tree model using the sklearn breast_cancer dataset.
    Optionally visualizes the data
    """
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Decision Tree Accuracy: {acc: .4f}")

    # Visualization
    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_classification_supervised

        # Reduce data to 2D (for visualization)
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)

        plot_classification_supervised(
            X_train_2d, X_test_2d, y_train, y_test, predictions,
            title="Decision Tree  Breast Cancer Classification\n(PCA-reduced)",
            xlabel="", ylabel="", labels=["Negative", "Positive"],
            supplimental_text=f"Accuracy: {acc:.3f}"
        )


if __name__ == '__main__':
    print('Testing Decision Tree algorithm')
    decision_tree(visualize=True)
