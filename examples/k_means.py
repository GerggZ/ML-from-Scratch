from sklearn.decomposition import PCA

from ml_from_scratch.algorithms import KMeans
from data_bases.get_database import get_sklearn_data_split


def k_means_example(
        X, y,
        visualize: bool = False
):
    """
    Example of how to use the K Means algorithm with the sklearn wine dataset
    Optionally visualizes the output using matplotlib
    """
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, max_iterations=300, distance_metric="euclidean")
    predictions = kmeans.fit_predict(X)

    print(f"\tK Means successfully clustered data\n\t(mark visualize=True to see plots)")

    if visualize:
        from examples.utils.plotting_regression_and_classification import plot_classification_unsupervised

        pca = PCA(n_components=2)  # Projecting data to 2D for visualization
        X_2d = pca.fit_transform(X)

        plot_classification_unsupervised(
            X_2d, y, predictions,
            ltitle="KMeans Clustering Results\n(PCA Reduced)", rtitle="Ground Truth Clustering\n(PCA Reduced)",
            xlabel="", ylabel=""
        )


if __name__ == '__main__':
    print('Testing K-Means Algorithm')
    train_data = get_sklearn_data_split("wine", random_state=42)
    k_means_example(*train_data, visualize=True)
