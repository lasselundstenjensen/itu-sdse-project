"""Image preprocessing and dataset-level feature artifacts."""

import json
from pathlib import Path

import numpy as np
from torchvision import transforms

from vial_classifier import config


def define_transforms():
    """Return (train_transforms, val_transforms); only training uses augmentation."""
    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.TRANSFORM_MEAN, std=config.TRANSFORM_STD),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.TRANSFORM_MEAN, std=config.TRANSFORM_STD),
        ]
    )

    return train_transforms, val_transforms


def calculate_image_statistics(data_loader):
    """Calculate mean and standard deviation per channel for a dataset."""
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]

        # Reshape to (batch_size, -1, 3) for channel-wise stats
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(batch_size, -1, 3)

        batch_mean = batch_images.mean(axis=1)
        batch_std = batch_images.std(axis=1)

        total_mean += batch_mean.sum(axis=0)
        total_std += batch_std.sum(axis=0)
        total_samples += batch_size

    overall_mean = (total_mean / total_samples).tolist()
    overall_std = (total_std / total_samples).tolist()

    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }


def save_class_names(path=config.CLASS_NAMES_PATH):
    """Save the class names as a JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.CLASS_NAMES, f, indent=2)
