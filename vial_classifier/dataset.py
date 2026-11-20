# Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import subprocess
import time

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vial_classifier.config import (
    BATCH_SIZE,
    IMAGE_DATA_DIR,
    IMAGE_SIZE,
    MAX_DOWNLOAD_WORKERS,
    METADATA_PATH,
    PROJ_ROOT,
    RANDOM_STATE,
    S3_BUCKET,
    S3_ENDPOINT,
    TEST_DATA_DIR,
    TEST_METADATA_PATH,
    TEST_RATIO,
    TRAIN_RATIO,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    VAL_RATIO,
)


def _setup_s3_client():
    """Create a boto3 S3 client configured for Hetzner's public bucket."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id='',
        aws_secret_access_key='',
        config=Config(signature_version=UNSIGNED),
    )



def _download_file_with_boto3(s3_client, key, output_path):
    """Download one image from S3, retrying if the request fails."""
    output_path = Path(output_path)

    if output_path.exists():
        return False

    for attempt in range(3):
        try:
            s3_client.download_file(S3_BUCKET, key, str(output_path))
            return True
        except Exception as error:
            if attempt < 2:
                print(f"Download failed for {key}, retrying...")
                time.sleep(2)
            else:
                print(f"Failed to download {key} after 3 attempts: {error}")
                return False


def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    s3_client = _setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(
                _download_file_with_boto3, s3_client, key, output_path
            )
            futures[future] = (key, output_path)

        for future in as_completed(futures):
            key, output_path = futures[future]
            if future.result():
                downloaded_count += 1
            else:
                if output_path.exists():
                    skipped_count += 1
                else:
                    failed_count += 1

    print(
        f"Downloaded {downloaded_count} images, "
        f"skipped {skipped_count} existing, failed {failed_count}"
    )
    if failed_count > 0:
        raise RuntimeError(f"Failed to download {failed_count} images")


def pull_data_from_dvc():
    """Pull the tracked metadata, then download the referenced images."""
    subprocess.run(["dvc", "pull"], cwd=PROJ_ROOT, check=True)

    download_files_from_metadata(
        METADATA_PATH,
        IMAGE_DATA_DIR,
        "images/raw/"
    )
    download_files_from_metadata(
        TEST_METADATA_PATH,
        TEST_DATA_DIR,
        "test/"
    )



def load_image_dataset():
    """Load (image_path, label) pairs from the raw images and the metadata."""
    print(f"Loading image dataset from {IMAGE_DATA_DIR}")
    print(f"Using metadata from {METADATA_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)
    print(f"Metadata loaded. Total entries: {len(metadata)}")

    image_dataset = []
    missing_files = []

    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = IMAGE_DATA_DIR / filename

        if image_path.exists():
            image_dataset.append((image_path, label))
        else:
            missing_files.append(filename)

    print(f"Loaded {len(image_dataset)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")

    return image_dataset



def define_transforms():
    print("Defining image transforms...")

    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    print("Training transforms: Resize, flip, rotation, jitter, ToTensor, Normalize")
    print("Validation transforms: Resize, ToTensor, Normalize")

    return train_transforms, val_transforms



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



def split_dataset(
    image_dataset,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_STATE
):
    """Stratified split so both classes keep the same balance in every set."""
    print(
        f"\nSplitting dataset into train ({train_ratio}), "
        f"val ({val_ratio}), test ({test_ratio})..."
    )

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]

    print(
        f"Class distribution before split: "
        f"Class 0: {len(class_0)}, Class 1: {len(class_1)}"
    )

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

    print(
        f"  Train - Class 0: {train_labels.count(0)}, "
        f"Class 1: {train_labels.count(1)}"
    )
    print(
        f"  Val   - Class 0: {val_labels.count(0)}, "
        f"Class 1: {val_labels.count(1)}"
    )
    print(
        f"  Test  - Class 0: {test_labels.count(0)}, "
        f"Class 1: {test_labels.count(1)}"
    )

    return train_dataset, val_dataset, test_dataset



def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=BATCH_SIZE
):
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")

    train_paths, train_labels = (
        zip(*train_dataset) if train_dataset else ([], [])
    )
    val_paths, val_labels = (
        zip(*val_dataset) if val_dataset else ([], [])
    )
    test_paths, test_labels = (
        zip(*test_dataset) if test_dataset else ([], [])
    )

    train_transforms, val_transforms = define_transforms()

    train_data = GlassVialDataset(
        list(train_paths),
        list(train_labels),
        train_transforms
    )
    val_data = GlassVialDataset(
        list(val_paths),
        list(val_labels),
        val_transforms
    )
    test_data = GlassVialDataset(
        list(test_paths),
        list(test_labels),
        val_transforms
    )

    # num_workers=0 keeps the DataLoader in-process (a class defined in a
    # notebook cannot be pickled to worker processes)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


