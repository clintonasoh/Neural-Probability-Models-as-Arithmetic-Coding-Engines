"""
download_data.py
----------------
Download Fashion MNIST and cache it as numpy arrays.

Fashion MNIST (Zalando Research, 2017):
  - 70,000 grayscale 28×28 images across 10 clothing categories
  - Train split: 60,000 images | Test split: 10,000 images
  - Drop-in replacement for MNIST; harder for simple classifiers

We store images as (N, 28, 28) uint8 arrays — raw pixel values [0, 255].
Labels are (N,) int64 arrays with values 0–9.

Usage:
    python download_data.py           # downloads and caches all splits
"""

import os
import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def download_fashion_mnist() -> None:
    """Download Fashion MNIST via torchvision and save as .npy files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError:
        raise ImportError("torchvision is required: pip install torchvision")

    print("[download] Fetching Fashion MNIST (torchvision)...")
    train_ds = FashionMNIST(DATA_DIR, train=True,  download=True)
    test_ds  = FashionMNIST(DATA_DIR, train=False, download=True)

    # torchvision already stores the raw uint8 tensors in .data
    for split, ds in [("train", train_ds), ("test", test_ds)]:
        images = ds.data.numpy().astype(np.uint8)    # (N, 28, 28)
        labels = ds.targets.numpy().astype(np.int64) # (N,)
        np.save(os.path.join(DATA_DIR, f"{split}_images.npy"), images)
        np.save(os.path.join(DATA_DIR, f"{split}_labels.npy"), labels)
        print(f"[download] {split:5s}: {images.shape}  → {DATA_DIR}/{split}_*.npy")


def load_fashion_mnist(split: str = "test"):
    """
    Load cached Fashion MNIST arrays.  Downloads automatically if not present.

    Parameters
    ----------
    split : 'train' or 'test'

    Returns
    -------
    images : np.ndarray (N, 28, 28) uint8
    labels : np.ndarray (N,)        int64
    """
    img_path = os.path.join(DATA_DIR, f"{split}_images.npy")
    lbl_path = os.path.join(DATA_DIR, f"{split}_labels.npy")

    if not os.path.exists(img_path):
        download_fashion_mnist()

    images = np.load(img_path)
    labels = np.load(lbl_path)
    return images, labels


if __name__ == "__main__":
    download_fashion_mnist()
    for split in ("train", "test"):
        imgs, lbls = load_fashion_mnist(split)
        print(f"{split}: images={imgs.shape} dtype={imgs.dtype}  "
              f"labels={lbls.shape}  classes={sorted(set(lbls.tolist()))}")
