import csv
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from botocore.config import Config
from botocore import UNSIGNED

from ..config import (
    ARTIFACT_DIR,
    IMAGE_DATA_DIR,
    METADATA_PATH,
    MLRUNS_DIR,
)

BASE_URL = "https://itudsip.hel1.your-objectstorage.com/data"

# S3 Configuration for Hetzner
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"

# Path configuration
IMAGE_DATA_DIR = Path("./data/images/raw")
METADATA_PATH = Path("./data/images/metadata.csv")
TEST_DATA_DIR = Path("./data/test")
TEST_METADATA_PATH = Path("./data/test/metadata.csv")


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
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False


def _get_s3_key_from_url(url):
    """Extract S3 key from a full URL.
    
    URL format: https://itudsip.hel1.your-objectstorage.com/data/images/metadata.csv
    Bucket: data
    Key: images/metadata.csv
    """
    # Remove the endpoint and the bucket name (first path component)
    key = url.replace(f"{S3_ENDPOINT}/{S3_BUCKET}/", "")
    return key


def download_with_dvc(url, out_path):
    """Download a single file using boto3 (DVC-compatible interface)."""
    s3_client = _setup_s3_client()
    out_path = Path(out_path)
    if out_path.exists():
        print(f"File {out_path} already exists, skipping download")
        return
    key = _get_s3_key_from_url(url)
    s3_client.download_file(S3_BUCKET, key, str(out_path))
    print(f"Downloaded {url} to {out_path}")


def download_files_from_metadata(metadata_url, output_dir, prefix=""):
    """
    Download files from S3 based on entries in metadata CSV.
    
    Args:
        metadata_url: URL to the metadata.csv file
        output_dir: Directory to save downloaded images
        prefix: Prefix for S3 object keys
    """
    s3_client = _setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract S3 key for metadata file
    metadata_key = _get_s3_key_from_url(metadata_url)
    
    # Download metadata to temp location
    temp_metadata = Path(output_dir) / "temp_metadata.csv"
    s3_client.download_file(S3_BUCKET, metadata_key, str(temp_metadata))
    
    # Read metadata CSV
    with open(temp_metadata, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Clean up temp file
    temp_metadata.unlink(missing_ok=True)
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row['filename']
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
    
    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")


def pull_data_from_dvc() -> None:
    Path("data/images").mkdir(parents=True, exist_ok=True)
    download_with_dvc(
        f"{BASE_URL}/images/metadata.csv",
        "data/images/metadata.csv"
    )
    download_files_from_metadata(
        f"{BASE_URL}/images/metadata.csv",
        "data/images/raw",
        "images/raw/"
    )
    
    Path("data/test").mkdir(parents=True, exist_ok=True)
    download_with_dvc(
        f"{BASE_URL}/test/metadata.csv",
        "data/test/metadata.csv"
    )
    download_files_from_metadata(
        f"{BASE_URL}/test/metadata.csv",
        "data/test",
        "test/"
    )


def setup_artifacts() -> None:
    """
    Create artifact and MLflow directories.

    Creates the following directories if they don't exist:
    - artifacts/ (for data and model artifacts)
    - mlruns/ (for MLflow tracking)
    - mlruns/.trash/ (for MLflow cleanup)
    - data/images/raw/ (for raw image storage)
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    (MLRUNS_DIR / ".trash").mkdir(parents=True, exist_ok=True)
    IMAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Created artifacts and image directories")


def load_image_dataset() -> List[Tuple[Path, int]]:
    """
    Load image dataset from IMAGE_DATA_DIR using metadata from METADATA_PATH.

    Reads images from the image directory and their corresponding labels
    from the metadata CSV file.

    Returns
    -------
    List[Tuple[Path, int]]
        List of (image_path, label) tuples where:
        - image_path: Path to the image file
        - label: Class label (0=good, 1=defective)

    Raises
    ------
    FileNotFoundError
        If METADATA_PATH or IMAGE_DATA_DIR does not exist.
    ValueError
        If metadata is empty or required columns are missing.
    """
    print(f"Loading image dataset from {IMAGE_DATA_DIR}")
    print(f"Using metadata from {METADATA_PATH}")

    # Check if metadata file exists
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. "
            "Create metadata.csv or ensure data exists."
        )

    # Check if image directory exists
    if not IMAGE_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found at {IMAGE_DATA_DIR}. "
            "Create the directory and add images."
        )

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Metadata loaded. Total entries: {len(metadata)}")

    # Build list of (image_path, label) tuples
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


def fetch_and_prepare_data() -> List[Tuple[Path, int]]:
    """
    Complete data fetching pipeline for image classification.

    Returns
    -------
    List[Tuple[Path, int]]
        Validated list of (image_path, label) tuples ready for preprocessing.
    """

    setup_artifacts()
    pull_data_from_dvc()

    # Load image dataset
    image_dataset = load_image_dataset()

    print(f"Data fetching complete. Valid images: {len(image_dataset)}")
    return image_dataset


if __name__ == "__main__":
    fetch_and_prepare_data()
