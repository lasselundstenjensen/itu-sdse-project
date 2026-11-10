"""Fetch the vial dataset from the object store and load it as (image_path, label) pairs."""

from concurrent.futures import ThreadPoolExecutor, as_completed #ThreadPoolExecutor lets you download several images simultaneously
import csv
from pathlib import Path
import subprocess
import sys

import boto3 
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd

from vial_classifier import config


def _setup_s3_client():
    """Create a boto3 S3 client for the public (unsigned) Hetzner bucket."""
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        config=Config(signature_version=UNSIGNED), #deleted access keys since it is a public bucket
    )


def _download_file_with_boto3(s3_client, key, output_path):
    """Download one file from the object storage. Returns True if downloaded, False if skipped or failed."""
    output_path = Path(output_path) #changes the string into a path
    if output_path.exists(): # avoids redownloading files
        return False
    try:
        s3_client.download_file(config.S3_BUCKET, key, str(output_path)) 
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False

# Deleted download_with_dvc and _get_s3_key_from_url since they were not called in the notebook


def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download every file listed in a metadata_CSV from the object store."""    
    # prefix tells the storage system where to find the image inside the bucket  
    s3_client = _setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f)) #Reads the entire CSV file and converts each row into a dictionary

    downloaded_count = skipped_count = failed_count = 0 #initializing all to 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(_download_file_with_boto3, s3_client, key, output_path)
            futures[future] =(key, output_path) #keeping both variables in case of failed downloads
        for future in as_completed(futures):
            if future.result():
                downloaded_count += 1
            elif output_path.exists():
                skipped_count += 1
            else:
                failed_count += 1

    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")


def pull_data_from_dvc():
    """Fetch the DVC-tracked metadata CSVs (DVC allows for reproduction of the experiment later) , then the training and test images they list."""
    subprocess.run([sys.executable, "-m", "dvc", "pull"], check=True)  #Runs a command in the operating system from inside Python
    download_files_from_metadata(config.METADATA_PATH, config.IMAGE_DATA_DIR, "images/raw/")
    download_files_from_metadata(config.TEST_METADATA_PATH, config.TEST_DATA_DIR, "test/")


def load_image_dataset(metadata_path=config.METADATA_PATH, image_data_dir=config.IMAGE_DATA_DIR):
    """Return a list of (image_path, label)/ python dataset for every image in the metadata that exists on disk."""
    print(f"Loading image dataset from {image_data_dir}")
    print(f"Using metadata from {metadata_path}")

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"Metadata loaded. Total entries: {len(metadata)}")

    image_dataset = []
    missing_files = []
    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = Path(image_data_dir) / filename
        if image_path.exists():
            image_dataset.append((image_path, label)) 
            missing_files.append(filename)

    print(f"Loaded {len(image_dataset)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")

    return image_dataset


if __name__ == "__main__":
    pull_data_from_dvc()
    load_image_dataset()