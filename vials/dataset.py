import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import typer
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, Dataset
import torch
import json
import yaml

from vials.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES_PATH,
    IMAGE_DATA_DIR,
    IMAGE_STATISTICS_PATH,
    METADATA_PATH,
    PROCESSED_DATA_DIR,
    TEST_DATA_DIR,
    TRAINING_CONFIG_PATH,
)
from vials.features import calculate_image_statistics, get_transforms

SPLIT_NAMES = ("train", "val", "test")

app = typer.Typer()

S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"

def _setup_s3_client():
    """Create a boto3 S3 client configured for Hetzner's public bucket."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id='',
        aws_secret_access_key='',
        config=Config(signature_version=UNSIGNED)
    )

def _download_file_with_boto3(s3_client, key, output_path):
    """Download a single file from S3 using boto3."""
    if output_path.exists():
        return False
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False

def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    s3_client = _setup_s3_client()
    output_dir.mkdir(parents=True, exist_ok=True)

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
            output_path = output_dir / filename
            future = executor.submit(_download_file_with_boto3, s3_client, key, output_path)
            futures[future] = output_path

        for future in as_completed(futures):
            output_path = futures[future]
            if future.result():
                downloaded_count += 1
            else:
                if output_path.exists():
                    skipped_count += 1
                else:
                    failed_count += 1

    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")

def pull_data_from_dvc():
    """Pull the training and test images listed in the metadata from the object store."""
    download_files_from_metadata(METADATA_PATH, IMAGE_DATA_DIR, "images/raw/")
    download_files_from_metadata(TEST_DATA_DIR / "metadata.csv", TEST_DATA_DIR, "test/")

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

def count_classes(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def print_section_header(title):
    """Pretty print a section header so the job log is readable."""
    logger.info("\n" + "=" * 50)
    logger.info(title)
    logger.info("=" * 50)

def dump_class_names():
    class_names = {0: "good", 1: "defective"}
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Class names saved to {}", CLASS_NAMES_PATH)

class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.copy()

        if self.transform:
            image = self.transform(image)

        return image, label

def split_dataset(image_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
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

def save_split(split, path):
    """Write (image_path, label) pairs as a filename,label csv relative to IMAGE_DATA_DIR."""
    pd.DataFrame(
        [(image_path.name, label) for image_path, label in split],
        columns=["filename", "label"],
    ).to_csv(path, index=False)
    logger.info("Saved {} rows to {}", len(split), path)

def load_split(path):
    """Read a split csv written by save_split back into (image_path, label) pairs."""
    df = pd.read_csv(path)
    return [(IMAGE_DATA_DIR / filename, int(label)) for filename, label in zip(df["filename"], df["label"])]

def save_splits(train_dataset, val_dataset, test_dataset, output_dir=PROCESSED_DATA_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in zip(SPLIT_NAMES, (train_dataset, val_dataset, test_dataset)):
        save_split(split, output_dir / f"{name}.csv")

def load_splits(input_dir=PROCESSED_DATA_DIR):
    """Load the train/val/test splits exported by this stage."""
    return tuple(load_split(input_dir / f"{name}.csv") for name in SPLIT_NAMES)

def load_batch_size():
    with open(TRAINING_CONFIG_PATH) as f:
        return yaml.safe_load(f)["training"]["batch_size"]

def save_image_statistics(train_loader):
    """Compute channel mean/std over the training loader and save them as an artifact."""
    image_stats = calculate_image_statistics(train_loader)
    with open(IMAGE_STATISTICS_PATH, "w") as f:
        json.dump(image_stats, f, indent=2)
    logger.info("Image statistics saved to {}", IMAGE_STATISTICS_PATH)

def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       train_transforms, val_transforms, batch_size=None):
    if batch_size is None:
        batch_size = load_batch_size()
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")

    train_paths, train_labels = zip(*train_dataset) if train_dataset else ([], [])
    val_paths, val_labels = zip(*val_dataset) if val_dataset else ([], [])
    test_paths, test_labels = zip(*test_dataset) if test_dataset else ([], [])

    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)

    # GPU Optimization, so only use when one is available
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


@app.command()
def main():
    subprocess.run(["dvc", "pull"])
    pull_data_from_dvc()

    dataset = load_image_dataset()

    labels = [label for _, label in dataset]
    class_dist = count_classes(labels)

    logger.info("Class distribution:")
    for cls, count in class_dist.items():
        logger.info("Class {}: {} images ({:.2f}%)", cls, count, count / len(labels) * 100)

    metadata_df = pd.read_csv(METADATA_PATH)
    logger.info("\nImages per class name: {}", metadata_df["class_name"].value_counts())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump_class_names()

    # COULD BE REMOVED AFTER TA ANSWER, BASICALLY DEAD-CODE
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    save_splits(train_dataset, val_dataset, test_dataset)

    train_transforms, val_transforms = get_transforms()
    train_loader, _, _ = create_dataloaders(
        train_dataset, val_dataset, test_dataset, train_transforms, val_transforms,
    )
    save_image_statistics(train_loader)


if __name__ == "__main__":
    app()
