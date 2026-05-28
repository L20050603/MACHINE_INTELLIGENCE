import numpy as np
from torchvision import datasets, transforms

from .mnist_downloader import ensure_mnist_available

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def to_one_hot(labels, num_classes=10):
    labels = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def load_mnist_numpy(data_dir="./data", limit_train=None, limit_test=None, val_size=5000, normalize=True):
    transform = transforms.ToTensor()
    ensure_mnist_available(data_dir, transform=transform)
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    x_train = np.asarray([img.numpy().reshape(-1) for img, _ in train_dataset], dtype=np.float32)
    y_train = np.asarray([label for _, label in train_dataset], dtype=np.int64)
    x_test = np.asarray([img.numpy().reshape(-1) for img, _ in test_dataset], dtype=np.float32)
    y_test = np.asarray([label for _, label in test_dataset], dtype=np.int64)

    if limit_train is not None:
        x_train = x_train[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test is not None:
        x_test = x_test[:limit_test]
        y_test = y_test[:limit_test]

    val_size = min(val_size, max(1, len(x_train) // 5))
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    if normalize:
        x_train = (x_train - MNIST_MEAN) / MNIST_STD
        x_val = (x_val - MNIST_MEAN) / MNIST_STD
        x_test = (x_test - MNIST_MEAN) / MNIST_STD

    return (
        x_train,
        to_one_hot(y_train),
        x_val,
        to_one_hot(y_val),
        x_test,
        to_one_hot(y_test),
    )
