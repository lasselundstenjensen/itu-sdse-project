"""Turn the (image_path, label) list into train/val/test DataLoaders, plus small dataset artifacts."""

import json
from pathlib import Path  

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vial_classifier import config  # [CHANGED] New: settings come from config.py instead of the past notebook globals


def calculate_image_statistics(data_loader):
    """Calculate mean and standard deviation per channel for a dataset."""
    print("Calculating image statistics...")

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

    print(f"Image statistics: Mean={overall_mean}, Std={overall_std}")
    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }


def save_class_names(path=config.CLASS_NAMES_PATH):
    # [CHANGED] New function wrapping code cell 10, which was loose code that ran as soon as the cell ran.
    """Save the label -> class name mapping as a JSON artifact."""
    class_names = {0: "good", 1: "defective"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # [CHANGED] mkdir added: the notebook created artifacts/ once in code cell 2; here each writer makes sure it exists.
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {path}")


def define_transforms():
    print("Defining image transforms...")

    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),  # [CHANGED] Was the notebook global image_size.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.TRANSFORM_MEAN, std=config.TRANSFORM_STD),
        # [CHANGED] Was the notebook globals transform_mean / transform_std.
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),  # [CHANGED] Was the notebook global image_size.
        transforms.ToTensor(),
        transforms.Normalize(mean=config.TRANSFORM_MEAN, std=config.TRANSFORM_STD),
        # [CHANGED] Was the notebook globals transform_mean / transform_std.
    ])

    print("Training transforms: Resize, flip, rotation, jitter, ToTensor, Normalize")
    print("Validation transforms: Resize, ToTensor, Normalize")
    return train_transforms, val_transforms

# [CHANGED] Removed `train_transforms, val_transforms = define_transforms()` that ran at the end of
# code cell 11. Those were globals used silently by create_dataloaders; it now calls define_transforms() itself.


class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # [CHANGED] Removed `from PIL import Image` here: moved to the top with the other imports.
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.copy()

        if self.transform:
            image = self.transform(image)

        return image, label


def split_dataset(image_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  random_seed=config.RANDOM_STATE):
    """Stratified split so both classes keep the same balance in every set."""
    print(f"\nSplitting dataset into train ({train_ratio}), val ({val_ratio}), test ({test_ratio})...")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]
    print(f"Class distribution before split: Class 0: {len(class_0)}, Class 1: {len(class_1)}")

    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        return train_data, val_data, test_data

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train_dataset = train_0 + train_1
    val_dataset = val_0 + val_1
    test_dataset = test_0 + test_1

    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    print("Split complete:")
    print(f"  Training set: {len(train_dataset)} images")
    print(f"  Validation set: {len(val_dataset)} images")
    print(f"  Test set: {len(test_dataset)} images")

    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    test_labels = [label for _, label in test_dataset]
    print(f"  Train - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"  Val   - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
    print(f"  Test  - Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=config.BATCH_SIZE):
    # [CHANGED] Default batch_size was the notebook global batch_size; now config.BATCH_SIZE.
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")

    train_transforms, val_transforms = define_transforms()
    # [CHANGED] New line: the notebook used the globals train_transforms / val_transforms
    # created at the end of code cell 11. Same transforms, no hidden globals.

    train_paths, train_labels = zip(*train_dataset) if train_dataset else ([], [])
    val_paths, val_labels = zip(*val_dataset) if val_dataset else ([], [])
    test_paths, test_labels = zip(*test_dataset) if test_dataset else ([], [])

    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def build_features(image_dataset):
    # [CHANGED] New function. It replaces the loose code at the end of code cells 10 and 14
    # (save class names, split, create loaders, save image stats) in the same order, so the
    # pipeline can run it with one call and pass the loaders on to train.py.
    """Notebook order: class names -> split -> loaders -> image statistics. Returns the 3 loaders."""
    save_class_names()

    train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    # Save the image statistics of the training set as an artifact
    image_stats = calculate_image_statistics(train_loader)
    Path(config.IMAGE_STATISTICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    # [CHANGED] mkdir added: the notebook created artifacts/ once in code cell 2.
    with open(config.IMAGE_STATISTICS_PATH, "w") as f:  # [CHANGED] Was the notebook global image_statistics_path.
        json.dump(image_stats, f, indent=2)
    print(f"Image statistics saved to {config.IMAGE_STATISTICS_PATH}")

    return train_loader, val_loader, test_loader