"""
Task 1: Data preparation for CIFAR-10
Implements data loading with augmentation (random horizontal flip, random crop, normalization)
"""
import os
import sys

# Add parent directory and current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import urllib.request
import tarfile

from config import CIFAR10_MEAN, CIFAR10_STD, RANDOM_CROP_SIZE, RANDOM_CROP_PADDING, BATCH_SIZE, DATA_ROOT


# Fast mirror URLs for CIFAR-10
MIRROR_URLS = [
    "https://mirrors.tuna.tsinghua.edu.cn/pytorch/datasets/cifar-10-python.tar.gz",
    "https://pjreddie.com/media/files/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]


def download_cifar10_from_mirror(root, max_retries=3):
    """
    Download CIFAR-10 dataset from fast mirrors.

    Args:
        root (str): Root directory to store the dataset
        max_retries (int): Maximum number of retries for each mirror
    """
    os.makedirs(root, exist_ok=True)

    # Check if dataset already exists
    base_folder = os.path.join(root, 'cifar-10-batches-py')
    if os.path.exists(base_folder):
        print(f"Dataset already exists at {base_folder}")
        return

    print("Downloading CIFAR-10 from mirror...")

    for url in MIRROR_URLS:
        for attempt in range(max_retries):
            try:
                filename = url.split('/')[-1]
                tar_path = os.path.join(root, filename)

                print(f"Trying: {url}")
                print(f"Saving to: {tar_path}")

                # Download with progress
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100.0 / total_size, 100) if total_size > 0 else 0
                    print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='')

                urllib.request.urlretrieve(url, tar_path, reporthook=show_progress)
                print("\nDownload complete!")

                # Extract
                print("Extracting...")
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=root)

                # Clean up tar file
                os.remove(tar_path)
                print(f"Dataset extracted to {base_folder}")
                return

            except Exception as e:
                print(f"\nFailed (attempt {attempt + 1}/{max_retries}): {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)

        print(f"Mirror {url} failed, trying next...")

    raise Exception("All mirrors failed!")


def get_train_transform():
    """
    Returns training data transformation pipeline with data augmentation.

    Augmentation includes:
    - Random horizontal flip (p=0.5)
    - Random crop after padding
    - ToTensor conversion
    - Normalization with CIFAR-10 mean and std

    Returns:
        transforms.Compose: Transformation pipeline
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(
            size=RANDOM_CROP_SIZE,
            padding=RANDOM_CROP_PADDING,
            padding_mode='reflect'
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


def get_test_transform():
    """
    Returns test data transformation pipeline (no augmentation).

    Only includes:
    - ToTensor conversion
    - Normalization with CIFAR-10 mean and std

    Returns:
        transforms.Compose: Transformation pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


def get_cifar10_dataloaders(batch_size=BATCH_SIZE, val_split=0.1, num_workers=2, data_dir=None):
    """
    Creates train, validation, and test dataloaders for CIFAR-10.

    Args:
        batch_size (int): Batch size for dataloaders
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of workers for data loading
        data_dir (str): Directory to store/load data (default: DATA_ROOT)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use provided data_dir or default
    data_root = data_dir if data_dir is not None else DATA_ROOT

    # Download from mirror first (much faster for servers in China)
    download_cifar10_from_mirror(data_root)

    # Load CIFAR-10 training dataset (already downloaded from mirror)
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,  # Already downloaded from mirror
        transform=get_train_transform()
    )

    # Split training data into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Load CIFAR-10 test dataset (already downloaded from mirror)
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,  # Already downloaded from mirror
        transform=get_test_transform()
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader = get_cifar10_dataloaders()

    # Check batch shape
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
