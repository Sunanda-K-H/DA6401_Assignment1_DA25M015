import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


def preprocess(x):
    x = x.astype("float32") / 255.0
    return x


def one_hot(y, num_classes=10):
    y = y.astype(int)
    return np.eye(num_classes)[y]


def load_dataset(name="mnist", val_size=0.1, random_state=42):
    if name == "mnist":
        data = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="liac-arff",
        )
    elif name == "fashion_mnist":
        data = fetch_openml(
            "Fashion-MNIST",
            version=1,
            as_frame=False,
            parser="liac-arff",
        )
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")
    X = data.data
    y = data.target.astype(int)

    X = preprocess(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=10000, stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_test = one_hot(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
