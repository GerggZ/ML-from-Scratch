from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml_from_scratch.algorithms import LinearRegression
from ml_from_scratch.utils.metrics import mean_squared_error


def linear_regression(visualize: bool = False) -> None:
    """
    Test the LinearRegression model on synthetic data using Mean Squared Error (MSE).

    Args:
        - visualize (bool): To plot/visualize test data using matplotlib
    """
    # Load data
    data = fetch_california_housing(as_frame=True)
    X = data.data[['MedInc']].to_numpy()
    y = data.target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Compute mean/std and transform
    X_test = scaler.transform(X_test)

    # Initialize and train the Linear Regression model
    reg = LinearRegression(learning_rate=0.01, n_iterations=1000)
    reg.fit(X_train, y_train)

    # Make class_predictions
    predictions = reg.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    print(f"Linear Regression Model trained successfully")
    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_linear_regression

        if X_test.shape[1] > 1:
            # Apply PCA to reduce to 1D for visualization
            pca = PCA(n_components=1)
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
    linear_regression(visualize=True)
