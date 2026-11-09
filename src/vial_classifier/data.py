"""Data acquisition, dataset construction and splitting.

Raw images and their metadata live in a public S3-compatible bucket. The
metadata CSVs are tracked with DVC; the images are downloaded individually
using the filenames listed in the metadata.
"""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

Sample = tuple[Path, int]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def make_s3_client():
    """Create a boto3 client for the public bucket (no credentials needed)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
    )


def s3_key_from_url(url: str) -> str:
    """Turn a public object URL into a bucket key, e.g. ``images/metadata.csv``."""
    prefix = f"{config.S3_ENDPOINT}/{config.S3_BUCKET}/"
    return url.replace(prefix, "")


def _download_one(s3_client, key: str, output_path: Path) -> bool:
    """Download one object. Returns True if it was fetched, False if skipped or failed."""
    if output_path.exists():
        return False
    try:
        s3_client.download_file(config.S3_BUCKET, key, str(output_path))
        return True
    except Exception as exc:  # noqa: BLE001 - keep going, count failures
        logger.warning("Failed to download %s: %s", key, exc)
        return False


def download_images_from_metadata(metadata_path: Path, output_dir: Path, prefix: str = "") -> dict[str, int]:
    """Download every file listed in a metadata CSV. Existing files are skipped."""
    s3_client = make_s3_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, newline="") as f:
        rows = list(csv.DictReader(f))

    downloaded = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=config.S3_DOWNLOAD_WORKERS) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            output_path = output_dir / filename
            future = executor.submit(_download_one, s3_client, f"{prefix}{filename}", output_path)
            futures[future] = output_path

        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            elif futures[future].exists():
                skipped += 1
            else:
                failed += 1

    logger.info("Downloaded %d images, skipped %d existing, failed %d", downloaded, skipped, failed)
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def pull_data() -> None:
    """Fetch the training and test images listed in their metadata files.

    The metadata CSVs themselves must already be present (``dvc pull``).
    """
    download_images_from_metadata(config.IMAGE_METADATA_PATH, config.RAW_IMAGE_DIR, "images/raw/")
    download_images_from_metadata(config.TEST_METADATA_PATH, config.TEST_DATA_DIR, "test/")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def load_image_dataset(
    metadata_path: Path = config.IMAGE_METADATA_PATH,
    image_dir: Path = config.RAW_IMAGE_DIR,
) -> list[Sample]:
    """Return ``(image_path, label)`` pairs for every image present on disk."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    samples: list[Sample] = []
    missing: list[str] = []
    for _, row in metadata.iterrows():
        image_path = image_dir / row["filename"]
        if image_path.exists():
            samples.append((image_path, int(row["class"])))
        else:
            missing.append(row["filename"])

    logger.info("Loaded %d valid images with labels", len(samples))
    if missing:
        logger.warning("%d image files not found", len(missing))
    return samples


def class_counts(labels) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def split_dataset(
    samples: list[Sample],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    random_seed: int = config.RANDOM_STATE,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified split so both classes keep the same balance in every set.

    The test set takes whatever remains of each class after the train and
    validation slices, which is ~15% for the default ratios.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    by_class = {
        0: [s for s in samples if s[1] == 0],
        1: [s for s in samples if s[1] == 1],
    }
    logger.info("Class distribution before split: Class 0: %d, Class 1: %d", len(by_class[0]), len(by_class[1]))

    train, val, test = [], [], []
    for label in (0, 1):
        data = by_class[label]
        np.random.shuffle(data)
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        train += data[:train_size]
        val += data[train_size:train_size + val_size]
        test += data[train_size + val_size:]

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    logger.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# Transforms and tensors
# ---------------------------------------------------------------------------
def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transforms, eval_transforms)``.

    Training adds light augmentation; evaluation only resizes and normalises.
    """
    normalize = transforms.Normalize(mean=config.TRANSFORM_MEAN, std=config.TRANSFORM_STD)
    train_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, eval_transforms


def build_eval_transform() -> transforms.Compose:
    """The inference-time transform (same as validation/test)."""
    return build_transforms()[1]


class VialDataset(Dataset):
    """Loads vial images from disk and applies a transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.image_paths[idx]) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = img.copy()
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def create_dataloaders(
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    batch_size: int = config.BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders. ``num_workers=0`` keeps loading in-process."""
    train_transforms, eval_transforms = build_transforms()

    def unzip(samples):
        if not samples:
            return [], []
        paths, labels = zip(*samples)
        return list(paths), list(labels)

    train_paths, train_labels = unzip(train_samples)
    val_paths, val_labels = unzip(val_samples)
    test_paths, test_labels = unzip(test_samples)

    train_loader = DataLoader(
        VialDataset(train_paths, train_labels, train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        VialDataset(val_paths, val_labels, eval_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        VialDataset(test_paths, test_labels, eval_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def calculate_image_statistics(data_loader: DataLoader) -> dict:
    """Mean and mean per-image standard deviation per RGB channel over a loader."""
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        batch = images.numpy()
        n = batch.shape[0]
        batch = batch.transpose(0, 2, 3, 1).reshape(n, -1, 3)
        total_mean += batch.mean(axis=1).sum(axis=0)
        total_std += batch.std(axis=1).sum(axis=0)
        total_samples += n

    return {
        "mean": (total_mean / total_samples).tolist(),
        "std": (total_std / total_samples).tolist(),
        "total_samples": total_samples,
    }
