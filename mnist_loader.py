"""
MNIST Data Loader
==================
Downloads and preprocesses the MNIST handwritten digit dataset.
Uses scikit-learn's fetch_openml — downloads ~170 MB on first run, then cached.

Normalization: pixel values [0, 255] → [-1, 1]  (background=-1, digit=+1)
Split: standard MNIST — 60,000 train / 10,000 test
"""

import numpy as np


def load_mnist():
    """
    Load the MNIST dataset.

    First call downloads ~170 MB from OpenML and caches locally.
    Subsequent calls load from cache (fast).

    Returns:
        X_train      : (60000, 784) float64, values in [-1, 1]
        X_test       : (10000, 784) float64, values in [-1, 1]
        y_train      : (60000, 10)  one-hot encoded labels
        y_test       : (10000, 10)  one-hot encoded labels
        y_train_labels: (60000,)    integer class labels 0-9
        y_test_labels : (10000,)    integer class labels 0-9
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError(
            "scikit-learn is required to load MNIST.\n"
            "Install it with:  pip install scikit-learn"
        )

    print("Loading MNIST dataset...")
    print("(First run downloads ~170 MB - this may take 1-2 minutes)")

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(int)

    # Normalize pixel values from [0, 255] to [-1, 1]
    X = X / 127.5 - 1.0

    # Standard MNIST split: first 60K train, last 10K test
    X_train, X_test = X[:60000], X[60000:]
    y_train_labels = y[:60000]
    y_test_labels = y[60000:]

    # One-hot encode labels
    y_train = np.eye(10)[y_train_labels]
    y_test = np.eye(10)[y_test_labels]

    print(f"Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")
    print(f"Input: {X_train.shape[1]} features (28x28 pixels)")
    print(f"Classes: digits 0-9")

    return X_train, X_test, y_train, y_test, y_train_labels, y_test_labels


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, y_trl, y_tel = load_mnist()
    print(f"\nPixel range (train): [{X_train.min():.2f}, {X_train.max():.2f}]")
    counts = dict(zip(*np.unique(y_trl, return_counts=True)))
    print(f"Samples per class (train): {counts}")
