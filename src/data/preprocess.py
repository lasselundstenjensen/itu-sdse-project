"""
Image Data Preprocessing Module

Handles image data preprocessing for the CNN model.
This includes defining transforms, creating PyTorch DataLoaders,
and splitting the dataset into train/val/test sets.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from ..config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGE_STATISTICS_PATH,
    NUM_CLASSES,
    RANDOM_STATE,
    TRAIN_RATIO,
    TEST_RATIO,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    VAL_RATIO,
)
from ..utils import (
    calculate_image_statistics,
    count_classes,
    print_section_header,
)


class GlassVialDataset(Dataset):
    """
    Custom PyTorch Dataset for glass vial images.

    Loads images and their labels for training and inference.
    Applies specified transforms to the images.

    Parameters
    ----------
    image_paths : List[Path]
        List of paths to image files.
    labels : List[int]
        List of corresponding class labels (0=good, 1=defective).
    transform : callable, optional
        Optional transform to be applied to images.
    """

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item at index idx.

        Returns
        -------
        Tuple[torch.Tensor, int]
            - Transformed image tensor (C, H, W) format
            - Class label
        """
        from PIL import Image

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.copy()

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label


def define_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Define PyTorch transforms for training and validation.

    Training transforms include data augmentation:
    - Random horizontal flip (50% probability)
    - Random rotation (±10 degrees)
    - Color jitter (brightness, contrast, saturation)
    - Resize to 64x64
    - Convert to tensor
    - Normalize using ImageNet statistics

    Validation transforms (no augmentation):
    - Resize to 64x64
    - Convert to tensor
    - Normalize using ImageNet statistics

    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        - Training transforms with augmentation
        - Validation transforms without augmentation
    """
    print("Defining image transforms...")

    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    print("Training transforms: Resize → RandomHorizontalFlip → RandomRotation → ColorJitter → ToTensor → Normalize")
    print("Validation transforms: Resize → ToTensor → Normalize")

    return train_transforms, val_transforms


def split_dataset(
    image_dataset: List[Tuple[Path, int]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_STATE,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Split dataset into train, validation, and test sets with stratified sampling.

    Ensures that each set maintains the same class distribution (50/50 balance).

    Parameters
    ----------
    image_dataset : List[Tuple[Path, int]]
        List of (image_path, label) tuples.
    train_ratio : float, default=TRAIN_RATIO from config
        Proportion of data for training.
    val_ratio : float, default=VAL_RATIO from config
        Proportion of data for validation.
    test_ratio : float, default=TEST_RATIO from config
        Proportion of data for testing.
    random_seed : int, default=RANDOM_STATE from config
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List, List, List]
        - Training set: List of (image_path, label) tuples
        - Validation set: List of (image_path, label) tuples
        - Test set: List of (image_path, label) tuples
    """
    print(f"\nSplitting dataset into train ({train_ratio}), val ({val_ratio}), test ({test_ratio})...")

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Separate by class for stratified sampling
    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]

    print(f"Class distribution before split: Class 0: {len(class_0)}, Class 1: {len(class_1)}")

    # Shuffle each class
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    # Split each class into train/val/test
    def split_class(data: List, ratios: Tuple[float, float, float]) -> Tuple[List, List, List]:
        total = len(data)
        train_size = int(total * ratios[0])
        val_size = int(total * ratios[1])
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data

    # Split each class
    train_0, val_0, test_0 = split_class(class_0, (train_ratio, val_ratio, test_ratio))
    train_1, val_1, test_1 = split_class(class_1, (train_ratio, val_ratio, test_ratio))

    # Combine classes for final datasets
    train_dataset = train_0 + train_1
    val_dataset = val_0 + val_1
    test_dataset = test_0 + test_1

    # Shuffle the combined datasets
    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    print(f"Split complete:")
    print(f"  Training set: {len(train_dataset)} images")
    print(f"  Validation set: {len(val_dataset)} images")
    print(f"  Test set: {len(test_dataset)} images")

    # Print class distribution for each set
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    test_labels = [label for _, label in test_dataset]

    print(f"\nClass distribution after split:")
    print(f"  Train - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"  Val - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
    print(f"  Test - Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: List[Tuple[Path, int]],
    val_dataset: List[Tuple[Path, int]],
    test_dataset: List[Tuple[Path, int]],
    batch_size: int = BATCH_SIZE,
    train_transforms: transforms.Compose = None,
    val_transforms: transforms.Compose = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader objects for training, validation, and testing.

    Parameters
    ----------
    train_dataset : List[Tuple[Path, int]]
        Training data as list of (image_path, label) tuples.
    val_dataset : List[Tuple[Path, int]]
        Validation data as list of (image_path, label) tuples.
    test_dataset : List[Tuple[Path, int]]
        Test data as list of (image_path, label) tuples.
    batch_size : int, default=BATCH_SIZE from config
        Batch size for DataLoaders.
    train_transforms : transforms.Compose, optional
        Transforms to apply to training data.
    val_transforms : transforms.Compose, optional
        Transforms to apply to validation and test data.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        - Training DataLoader (shuffled)
        - Validation DataLoader
        - Test DataLoader
    """
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")

    # Extract image paths and labels
    train_paths, train_labels = zip(*train_dataset) if train_dataset else ([], [])
    val_paths, val_labels = zip(*val_dataset) if val_dataset else ([], [])
    test_paths, test_labels = zip(*test_dataset) if test_dataset else ([], [])

    # If no transforms provided, define default ones
    if train_transforms is None or val_transforms is None:
        train_transforms, val_transforms = define_transforms()

    # Create datasets
    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=2,
        pin_memory=True,
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def preprocess_data(
    image_dataset: List[Tuple[Path, int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Complete image data preprocessing pipeline.

    This function combines all preprocessing steps in sequence:
    1. Define transforms for data augmentation
    2. Split dataset into train/val/test sets with stratified sampling
    3. Create PyTorch DataLoader objects
    4. Save preprocessing metadata (class distribution, image statistics)

    Parameters
    ----------
    image_dataset : List[Tuple[Path, int]], optional
        List of (image_path, label) tuples. If None, loads from fetch_and_prepare_data().

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        - Training DataLoader
        - Validation DataLoader
        - Test DataLoader
    """
    print_section_header("IMAGE DATA PREPROCESSING")

    # Load image dataset if not provided
    if image_dataset is None:
        from .fetch import fetch_and_prepare_data
        image_dataset = fetch_and_prepare_data()

    # Check that we have data
    if not image_dataset:
        raise ValueError("No images found in dataset. Check data loading step.")

    print(f"Starting preprocessing with {len(image_dataset)} images")

    # Print initial class distribution
    labels = [label for _, label in image_dataset]
    class_dist = count_classes(labels)
    print(f"\nInitial class distribution:")
    for cls, count in class_dist.items():
        print(f"  Class {cls}: {count} images ({count/len(labels)*100:.1f}%)")

    # Define transforms
    train_transforms, val_transforms = define_transforms()

    # Split dataset into train/val/test
    train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        train_transforms=train_transforms, 
        val_transforms=val_transforms
    )

    # Calculate and save image statistics from training set
    print(f"\nCalculating image statistics from training set...")
    image_stats = calculate_image_statistics(train_loader)
    IMAGE_STATISTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_STATISTICS_PATH, "w") as f:
        json.dump(image_stats, f, indent=2)
    print(f"Image statistics saved to {IMAGE_STATISTICS_PATH}")

    print(f"\nPreprocessing complete.")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    Run image data preprocessing module directly.

    This will load images from the default location and run preprocessing.

    Usage:
        python -m src.data.preprocess
    """
    print("Running image data preprocessing module...")
    try:
        # Load and preprocess data
        train_loader, val_loader, test_loader = preprocess_data()
        print(f"\nPreprocessing complete.")
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader: {len(val_loader)} batches")
        print(f"  Test loader: {len(test_loader)} batches")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
