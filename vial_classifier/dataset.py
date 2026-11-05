"""Data acquisition and loading for the vial images.

CLI: `python -m vial_classifier.dataset` pulls the DVC-tracked metadata, then downloads
the training and test images from the object store.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import subprocess
import sys

import boto3
from boto3.exceptions import Boto3Error
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import typer

from vial_classifier import config

app = typer.Typer()


def get_s3_client():
    """Create a boto3 S3 client configured for Hetzner's public bucket."""
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
    )


def _download_file_with_boto3(s3_client, key, output_path):
    """Download a single file, returning True only if it was actually downloaded."""
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(config.S3_BUCKET, key, str(output_path))
        return True
    except (Boto3Error, BotoCoreError, ClientError, OSError) as e:
        logger.error(f"Failed to download {key}: {e}")
        return False


def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    s3_client = get_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(_download_file_with_boto3, s3_client, key, output_path)
            futures[future] = output_path

        for future in as_completed(futures):
            if future.result():
                downloaded_count += 1
            elif futures[future].exists():
                skipped_count += 1
            else:
                failed_count += 1

    logger.info(
        f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, "
        f"failed {failed_count}"
    )
    if failed_count:
        # load_image_dataset skips missing files, so continuing would train on less data
        raise RuntimeError(f"{failed_count} images failed to download into {output_dir}")


def pull_metadata():
    """Fetch the DVC-tracked metadata CSVs."""
    subprocess.run([sys.executable, "-m", "dvc", "pull"], check=True, cwd=config.PROJ_ROOT)


def load_image_dataset(metadata_path=config.TRAIN_METADATA_PATH, image_dir=config.TRAIN_DATA_DIR):
    """Load (image_path, label) pairs from the raw images and the metadata."""
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    image_dataset = []
    missing_files = []
    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = Path(image_dir) / filename
        if image_path.exists():
            image_dataset.append((image_path, label))
        else:
            missing_files.append(filename)

    logger.info(f"Loaded {len(image_dataset)} of {len(metadata)} images from {image_dir}")
    if missing_files:
        logger.warning(f"{len(missing_files)} image files not found")

    class_counts = Counter(label for _, label in image_dataset)
    for label, count in sorted(class_counts.items()):
        logger.info(f"Class {label}: {count} images ({count / len(image_dataset):.1%})")

    return image_dataset


def split_dataset(
    image_dataset,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    random_seed=config.RANDOM_STATE,
):
    """Stratified split so both classes keep the same balance in every set.

    The seeds and the order of the shuffle calls must match the notebook: changing either
    changes the split and every random number consumed after it.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]

    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size + val_size :]
        return train_data, val_data, test_data

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train_dataset = train_0 + train_1
    val_dataset = val_0 + val_1
    test_dataset = test_0 + test_1

    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    logger.info(
        f"Split: train={len(train_dataset)} (class 0: {len(train_0)}, class 1: {len(train_1)}), "
        f"val={len(val_dataset)} (class 0: {len(val_0)}, class 1: {len(val_1)}), "
        f"test={len(test_dataset)} (class 0: {len(test_0)}, class 1: {len(test_1)})"
    )

    return train_dataset, val_dataset, test_dataset


class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = img.copy()

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    train_transforms,
    val_transforms,
    batch_size=config.BATCH_SIZE,
):
    """Build the DataLoaders; only the training set is shuffled and augmented."""
    train_paths, train_labels = zip(*train_dataset) if train_dataset else ([], [])
    val_paths, val_labels = zip(*val_dataset) if val_dataset else ([], [])
    test_paths, test_labels = zip(*test_dataset) if test_dataset else ([], [])

    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)

    # num_workers=0 is kept for parity with the notebook's random number sequence
    loader_args = {"batch_size": batch_size, "num_workers": 0, "pin_memory": True}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    logger.info(
        f"DataLoaders (batch_size={batch_size}): train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader


@app.command()
def main():
    """Pull the metadata, then download the training and test images."""
    pull_metadata()
    download_files_from_metadata(
        config.TRAIN_METADATA_PATH, config.TRAIN_DATA_DIR, config.S3_TRAIN_PREFIX
    )
    download_files_from_metadata(
        config.TEST_METADATA_PATH, config.TEST_DATA_DIR, config.S3_TEST_PREFIX
    )


if __name__ == "__main__":
    app()
