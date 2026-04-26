import gzip
import shutil
import ssl
import urllib.request
from pathlib import Path

from torchvision import datasets


MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def _raw_dir(data_dir):
    return Path(data_dir) / "MNIST" / "raw"


def _uncompressed_name(gzip_name):
    return gzip_name[:-3]


def _download_file(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        with open(target, "wb") as f:
            shutil.copyfileobj(response, f)


def _extract_gzip(source, target):
    with gzip.open(source, "rb") as gz:
        with open(target, "wb") as out:
            shutil.copyfileobj(gz, out)


def _ensure_raw_files(data_dir):
    raw = _raw_dir(data_dir)
    raw.mkdir(parents=True, exist_ok=True)

    for filename in MNIST_FILES:
        gzip_path = raw / filename
        raw_path = raw / _uncompressed_name(filename)

        if not raw_path.exists():
            if not gzip_path.exists():
                last_error = None
                for mirror in MIRRORS:
                    url = mirror + filename
                    try:
                        print(f"Downloading MNIST fallback file: {url}")
                        _download_file(url, gzip_path)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    raise RuntimeError(f"Could not download {filename}: {last_error}") from last_error

            print(f"Extracting MNIST file: {gzip_path.name}")
            _extract_gzip(gzip_path, raw_path)


def ensure_mnist_available(data_dir, transform=None):
    """Make torchvision MNIST robust against temporary mirror or SSL failures."""

    try:
        datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    except RuntimeError as exc:
        print(f"torchvision MNIST download failed, using fallback downloader: {exc}")
        _ensure_raw_files(data_dir)

