from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml_from_scratch.algorithms import LinearRegression
from ml_from_scratch.utils.metrics import mean_squared_error
from data_bases.get_database import get_sklearn_data_split


def linear_regression_example(
        X_train, X_test, y_train, y_test,
        visualize: bool = False
):
    """
    Example of how to use the Linear Regression algorithm with the sklearn california housing dataset
    Optionally visualizes the output using matplotlib
    """
    reg = LinearRegression(learning_rate=0.01, n_iterations=1000)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"\tLinear Regression Mean Squared Error: {mse:.4f}")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_linear_regression

        if X_test.shape[1] > 1:
            pca = PCA(n_components=1)  # Projecting data to 1D for visualization
            pca.fit(X_train)
            X_test = pca.transform(X_test)

        plot_linear_regression(
            X_test, y_test, predictions,
            title="Linear Regression on California Housing",
            xlabel="Median Income", ylabel="House Price",
            supplimental_text=f"MSE: {mse:.4f}"
        )


if __name__ == '__main__':
    print('Testing Linear Regression algorithm')
    data = fetch_california_housing(as_frame=True)
    X, y = data.data[['MedInc']].to_numpy(), data.target.to_numpy()
    X = StandardScaler().fit_transform(X)

    # Remove housing prices over 5, because they seem....just off
    mask = y <= 4.99
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    linear_regression_example(X_train, X_test, y_train, y_test, visualize=True)
