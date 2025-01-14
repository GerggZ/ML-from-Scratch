from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_from_scratch.algorithms import KNearestNeighbors


def test_k_nearest_neighbors(visualize: bool = False):
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

    # Make class_predictions
    train_predictions = cls.predict(X=X_train)
    test_predictions = cls.predict(X=X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)