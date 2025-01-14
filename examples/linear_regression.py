from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from ml_from_scratch.algorithms import LinearRegression
from ml_from_scratch.utils.metrics import mean_squared_error


def test_linear_regression(visualize: bool = False) -> None:
    """
    Test the LinearRegression model on synthetic data using Mean Squared Error (MSE).

    Args:
        - visualize (bool): To plot/visualize test data using matplotlib
    """
    # Create synthetic regression data
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Initialize and train the Linear Regression model
    reg = LinearRegression(learning_rate=0.01, num_iterations=1000)
    reg.fit(X_train, y_train)

    # Make class_predictions
    predictions = reg.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    print(f"Linear Regression Model trained successfully")
    if visualize:
        import matplotlib.pyplot as plt

        # Get regression equation parameters
        slope = reg.weights[0]
        intercept = reg.bias
        equation = f"targets = {slope:.2f}x + {intercept:.2f}"

        # Visualize the result
        y_pred_line = reg.predict(X)
        cmap = plt.get_cmap('viridis')

        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label="Train Data")
        plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label="Test Data")
        plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")

        # Add labels, title, and legend
        plt.xlabel("Feature (features)", fontsize=12)
        plt.ylabel("Target (targets)", fontsize=12)
        plt.title("Linear Regression Test", fontsize=14)
        plt.legend()

        # Display MSE and equation on the graph
        plt.text(
            0.05, 0.95,
            f"MSE: {mse:.2f}\n{equation}",
            fontsize=10,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
        )

        plt.show()

