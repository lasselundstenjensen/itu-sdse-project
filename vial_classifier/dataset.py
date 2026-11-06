"""Fetch the vial dataset from the object store and load it as (image_path, label) pairs."""

import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from vial_classifier import config


def _setup_s3_client():
    """Create a boto3 S3 client for the public (unsigned) Hetzner bucket."""
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT, # [CHANGED] Was the global S3_ENDPOINT. Points at Hetzner instead of AWS.
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
    )


def _download_file_with_boto3(s3_client, key, output_path):
    """Download one file. Returns True if downloaded, False if skipped or failed."""
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(config.S3_BUCKET, key, str(output_path)) # [CHANGED] config.S3_BUCKET instead of the global.
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False


def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download every file listed in a metadata CSV from the object store."""
    s3_client = _setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))

    downloaded_count = skipped_count = failed_count = 0  # [CHANGED] Was 3 separate lines. Three counters, all starting at 0.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(_download_file_with_boto3, s3_client, key, output_path)
            futures[future] = output_path # [CHANGED] key variable dropped; key was never used afterwards.

        for future in as_completed(futures):
            if future.result():
                downloaded_count += 1
            elif futures[future].exists():
                skipped_count += 1
            else:
                failed_count += 1

    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")


def pull_data_from_dvc():
    """Fetch the DVC-tracked metadata CSVs, then the training and test images they list."""
    subprocess.run([sys.executable, "-m", "dvc", "pull"], check=True)
    download_files_from_metadata(config.METADATA_PATH, config.IMAGE_DATA_DIR, "images/raw/")
    download_files_from_metadata(config.TEST_METADATA_PATH, config.TEST_DATA_DIR, "test/")


def load_image_dataset(metadata_path=config.METADATA_PATH, image_data_dir=config.IMAGE_DATA_DIR):
    """Return a list of (image_path, label) for every image in the metadata that exists on disk."""
    print(f"Loading image dataset from {image_data_dir}")
    print(f"Using metadata from {metadata_path}")

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"Metadata loaded. Total entries: {len(metadata)}")

    image_dataset = []
    missing_files = []
    for _, row in metadata.iterrows():
        image_path = Path(image_data_dir) / row["filename"]
        if image_path.exists():
            image_dataset.append((image_path, int(row["class"]))) # [CHANGED] The notebook first stored filename and label in variables; now it is inlined here
        else:
            missing_files.append(row["filename"])

    print(f"Loaded {len(image_dataset)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")

    return image_dataset


if __name__ == "__main__":
    pull_data_from_dvc()
    load_image_dataset()