import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dvc.repo import Repo
import boto3
from botocore.config import Config
from botocore import UNSIGNED
import pandas as pd
import os

# READ DATA

PROJECT_ROOT = Path(__file__).resolve().parent.parent

metadata_path = "./data/images/metadata.csv"
test_data_dir = "./data/test"
image_data_dir = "./data/images/raw"

# S3 Configuration for Hetzner
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"

BASE_URL = "https://itudsip.hel1.your-objectstorage.com/data"

# Paths
#metadata_path = "./data/images/metadata.csv"
#image_data_dir = "./data/images/raw"

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
    # URL format: https://itudsip.hel1.your-objectstorage.com/data/images/metadata.csv
    # Bucket: data, Key: images/metadata.csv
    return url.replace(f"{S3_ENDPOINT}/{S3_BUCKET}/", "")

def download_with_dvc(url, out_path):
    """Download a single file (DVC-compatible interface)."""
    s3_client = _setup_s3_client()
    out_path = Path(out_path)
    if out_path.exists():
        print(f"File {out_path} already exists, skipping download")
        return
    key = _get_s3_key_from_url(url)
    s3_client.download_file(S3_BUCKET, key, str(out_path))
    print(f"Downloaded {url} to {out_path}")

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

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(_download_file_with_boto3, s3_client, key, output_path)
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

def pull_data_from_dvc():
    images_meta = PROJECT_ROOT / "data" / "images" / "metadata.csv"
    test_meta = PROJECT_ROOT / "data" / "test" / "metadata.csv"

    # Only pull if metadata.csv is completely missing
    if not images_meta.exists():
        try:
            repo = Repo(str(PROJECT_ROOT))
            repo.pull()
        except Exception as e:
            print(f"DVC pull skipped/failed: {e}")

    download_files_from_metadata(
        images_meta,
        PROJECT_ROOT / "data" / "images" / "raw",
        "images/raw/"
    )

    if test_meta.exists():
        download_files_from_metadata(
            test_meta,
            PROJECT_ROOT / "data" / "test",
            "test/"
)
     
# LOAD DATASET

def load_image_dataset(
    image_data_dir=PROJECT_ROOT / "data" / "images" / "raw",
    metadata_path=PROJECT_ROOT / "data" / "images" / "metadata.csv"
):
    """Load (image_path, label) pairs from the raw images and the metadata."""
    
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
        else:
            missing_files.append(filename)

    print(f"Loaded {len(image_dataset)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")

    return image_dataset

def get_image_dataset():
    """ Pulls data from DVC and returns the loaded image dataset"""
    pull_data_from_dvc()
    return load_image_dataset()

# The metadata csvs are tracked with DVC (see data/images/metadata.csv.dvc and
# data/test/metadata.csv.dvc), so dvc pull fetches them before we download the images.
if __name__ == "__main__":
    image_dataset = get_image_dataset()
    print("\nFirst 5 items in metadata")
    print(pd.read_csv(metadata_path).head(5))
    
    print("\nFirst 5 items in iamge dataset")
    for path, label in image_dataset[:5]:
        print(f"Path: {path} | Label: {label}")
