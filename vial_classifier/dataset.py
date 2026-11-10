from pathlib import Path
import typer
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import boto3
import typer
from botocore import UNSIGNED
from botocore.config import Config

from vial_classifier.config import (
    IMAGE_DIR,
    METADATA_PATH,
    S3_BUCKET,
    S3_ENDPOINT,
    TEST_DATA_DIR,
    TEST_METADATA_PATH,
)

app = typer.Typer()


def _setup_s3_client():
    """Create a boto3 S3 client configured for Hetzner's public bucket."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
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

    print(
        f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, "
        f"failed {failed_count}"
    )


def pull_data():
    """Download the training and test images listed in the metadata from the object store."""
    download_files_from_metadata(METADATA_PATH, IMAGE_DIR, "images/raw/")
    download_files_from_metadata(TEST_METADATA_PATH, TEST_DATA_DIR, "test/")


@app.command()
def main():
    """Download training and test images. Run `dvc pull` first to get the metadata."""
    pull_data()


if __name__ == "__main__":
    app()
