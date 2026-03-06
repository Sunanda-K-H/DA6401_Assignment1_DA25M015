from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess(x):
    n = x.shape[0]
    return x.reshape(n, -1).astype("float32") / 255.0


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def load_dataset(name="mnist", val_size=0.1, random_state=42):
    if name == "mnist":
        (x_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (x_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    x_train_full = preprocess(x_train_full)
    X_test = preprocess(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_test = one_hot(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
