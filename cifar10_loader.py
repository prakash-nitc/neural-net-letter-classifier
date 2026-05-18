"""
CIFAR-10 Data Loader
=====================
Downloads the CIFAR-10 dataset on first call (~163 MB), then caches locally.
CIFAR-10: 60,000 colour images (32x32 RGB) across 10 classes.
          50,000 train  /  10,000 test.

Preprocessing:
  - Flatten each image: 32x32x3 = 3072 features
  - Normalize per-channel using training-set mean and std
  - One-hot encode labels
"""

import os
import pickle
import tarfile
import ssl
import urllib.request

import numpy as np


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

_URL      = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CACHE    = os.path.join(os.path.expanduser('~'), '.cache', 'cifar10')
_DATA_DIR = os.path.join(_CACHE, 'cifar-10-batches-py')


def _download():
    os.makedirs(_CACHE, exist_ok=True)
    tar_path = os.path.join(_CACHE, 'cifar-10-python.tar.gz')

    if not os.path.isdir(_DATA_DIR):
        print('Downloading CIFAR-10 (~163 MB) ...')
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(_URL, context=ctx) as resp, \
             open(tar_path, 'wb') as f:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            chunk = 1 << 16   # 64 KB
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded / total * 100
                    print(f'\r  {pct:5.1f}%  ({downloaded >> 20} / {total >> 20} MB)',
                          end='', flush=True)
        print()

        print('Extracting ...')
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(_CACHE)
        print('Done.')


def _load_batch(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'].astype(np.float64), np.array(d[b'labels'])


def load_cifar10():
    """
    Download (if needed) and return CIFAR-10 train / test splits.

    Returns:
        X_train       : (50000, 3072)  float64, normalised
        X_test        : (10000, 3072)  float64, normalised
        y_train       : (50000, 10)    one-hot encoded
        y_test        : (10000, 10)    one-hot encoded
        y_train_labels: (50000,)       integer 0-9
        y_test_labels : (10000,)       integer 0-9
        class_names   : list of 10 strings
    """
    _download()

    # Load all five training batches
    X_tr_list, y_tr_list = [], []
    for i in range(1, 6):
        X, y = _load_batch(os.path.join(_DATA_DIR, f'data_batch_{i}'))
        X_tr_list.append(X)
        y_tr_list.append(y)

    X_train        = np.vstack(X_tr_list)
    y_train_labels = np.concatenate(y_tr_list)

    X_test, y_test_labels = _load_batch(os.path.join(_DATA_DIR, 'test_batch'))

    # Per-channel normalisation (zero mean, unit variance using training stats)
    # Each image is stored as [R*1024, G*1024, B*1024]
    channel_mean = X_train.reshape(-1, 3, 1024).mean(axis=(0, 2))   # shape (3,)
    channel_std  = X_train.reshape(-1, 3, 1024).std(axis=(0, 2))    # shape (3,)

    def _normalise(X):
        X = X.reshape(-1, 3, 1024)
        X = (X - channel_mean[:, None]) / (channel_std[:, None] + 1e-8)
        return X.reshape(-1, 3072)

    X_train = _normalise(X_train)
    X_test  = _normalise(X_test)

    y_train = np.eye(10)[y_train_labels]
    y_test  = np.eye(10)[y_test_labels]

    print(f'CIFAR-10 ready: {X_train.shape[0]:,} train  |  {X_test.shape[0]:,} test')
    print(f'Input: {X_train.shape[1]} features (32x32x3 pixels, per-channel normalised)')
    print(f'Classes: {", ".join(CLASSES)}')

    return X_train, X_test, y_train, y_test, y_train_labels, y_test_labels, CLASSES


if __name__ == '__main__':
    X_tr, X_te, y_tr, y_te, ytrl, ytel, cls = load_cifar10()
    print(f'X_train range: [{X_tr.min():.2f}, {X_tr.max():.2f}]  mean={X_tr.mean():.4f}')
