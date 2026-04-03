import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    CIFAR100_CLASS_NAMES,
    CIFAR100_MEAN,
    CIFAR100_STD,
    DATA_DIR,
    IMG_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SPLIT_SEED,
    TRAIN_RANDOM_CROP_PADDING,
    VAL_SPLIT,
)

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE, padding=TRAIN_RANDOM_CROP_PADDING),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ]
)

VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ]
)


def get_cifar100_class_names(data_dir=None, download: bool = False):
    """
    Fine label names in index order (same as torchvision CIFAR-100).

    Default ``download=False`` uses bundled names so inference works without the dataset.
    Set ``download=True`` to read names from disk after ``get_dataloaders`` (or Kaggle) has
    populated ``data_dir``.
    """
    if not download:
        return list(CIFAR100_CLASS_NAMES)
    root = DATA_DIR if data_dir is None else data_dir
    ds = datasets.CIFAR100(root=root, train=True, download=True)
    return ds.classes


def get_dataloaders(
    data_dir=None,
    batch_size=None,
    val_split=None,
    num_workers=None,
    pin_memory=None,
    split_seed=None,
):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    val_split = VAL_SPLIT if val_split is None else val_split
    num_workers = NUM_WORKERS if num_workers is None else num_workers
    pin_memory = PIN_MEMORY if pin_memory is None else pin_memory
    split_seed = SPLIT_SEED if split_seed is None else split_seed
    data_dir = DATA_DIR if data_dir is None else data_dir
    full_train_aug = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=TRAIN_TRANSFORMS
    )
    full_train_val = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=VAL_TRANSFORMS
    )
    test_set = datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=VAL_TRANSFORMS
    )

    n_val = int(len(full_train_aug) * val_split)
    n_train = len(full_train_aug) - n_val
    generator = torch.Generator().manual_seed(split_seed)
    train_idx, val_idx = random_split(
        range(len(full_train_aug)), [n_train, n_val], generator=generator
    )

    train_set = Subset(full_train_aug, train_idx.indices)
    val_set = Subset(full_train_val, val_idx.indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, 100, full_train_aug.classes
