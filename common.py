import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Primary mirror first, GitHub raw as fallback (same file names)
datasets.FashionMNIST.mirrors = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
]

DATA_DIR = os.path.expanduser("~/dl-lab/data")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530          # Fashion-MNIST training-set pixel statistics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def build_transforms(augment: bool = False):
    t = []
    if augment:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(28, padding=2)]
    t += [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    return transforms.Compose(t)


def get_datasets(augment: bool = False, val_size: int = 5000, seed: int = 42):
    train_aug = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
                                      transform=build_transforms(augment))
    train_clean = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
                                        transform=build_transforms(False))
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True,
                                 transform=build_transforms(False))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    return Subset(train_aug, train_idx), Subset(train_clean, val_idx), test


def get_loaders(batch_size: int = 128, augment: bool = False,
                train_subset: int | None = None, seed: int = 42):
    train_ds, val_ds, test_ds = get_datasets(augment=augment, seed=seed)
    if train_subset:
        train_ds = Subset(train_ds, list(range(train_subset)))
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


def get_clean_train_loader(train_subset: int | None = None, batch_size: int = 256, seed: int = 42):
    """Training images WITHOUT augmentation, used to measure the true training accuracy."""
    train_ds, _, _ = get_datasets(augment=False, seed=seed)
    if train_subset:
        train_ds = Subset(train_ds, list(range(train_subset)))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=False)
