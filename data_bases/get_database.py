from sklearn.datasets import (
    load_breast_cancer, load_digits, load_iris,
    load_diabetes, load_wine
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sklearn_dataset_map = {
    'breast cancer': load_breast_cancer,
    'digits': load_digits,
    'iris': load_iris,
    'diabetes': load_diabetes,
    'wine': load_wine
}


def get_sklearn_data_split(
        sklearn_dataset_name: str,
        test_size=None, train_size=None,
        use_standard_scaler=True, random_state=42
):
    data = sklearn_dataset_map[sklearn_dataset_name]()
    X, y = data.data, data.target

    if use_standard_scaler:
        X = StandardScaler().fit_transform(X)

    if test_size is not None or train_size is not None:
        return train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=random_state)
    return X, y