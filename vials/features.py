import numpy as np
import yaml
from torchvision import transforms

from vials.config import TRAINING_CONFIG_PATH


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

    print(f"Image statistics: Mean={overall_mean}, Std={overall_std}")
    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }

def load_image_specs():
    with open(TRAINING_CONFIG_PATH) as f:
        return yaml.safe_load(f)["image_specs"]

def define_transforms(image_size, transform_mean, transform_std):
    print("Defining image transforms...")

    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std),
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std),
    ])

    print("Training transforms: Resize, flip, rotation, jitter, ToTensor, Normalize")
    print("Validation transforms: Resize, ToTensor, Normalize")
    return train_transforms, val_transforms

def get_transforms():
    """Build train/val transforms from the image specs in the training config."""
    image_specs = load_image_specs()
    return define_transforms(
        image_specs["image_size"],
        image_specs["transform_mean"],
        image_specs["transform_std"],
    )
