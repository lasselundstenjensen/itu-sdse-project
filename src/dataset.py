"""Download the vial images from the object store (data ingestion step).

The metadata csvs (which image is clean/defective) are versioned with DVC,
so run `dvc pull` first. This script then downloads every image listed in
them from the public Hetzner bucket into data/images/raw and data/test.

Run it with:  uv run python -m src.dataset
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from loguru import logger
import pandas as pd
import typer

from src.config import (
    METADATA_PATH,
    RAW_IMAGES_DIR,
    S3_BUCKET,
    S3_ENDPOINT,
    TEST_DATA_DIR,
    TEST_METADATA_PATH,
)

app = typer.Typer()


def make_s3_client():
    """S3 client for the public bucket (no credentials needed)."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
    )


def download_file(s3_client, key: str, output_path: Path) -> str:
    """Download one file. Returns "downloaded", "skipped" (already there) or "failed"."""
    if output_path.exists():
        return "skipped"
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return "downloaded"
    except Exception as e:
        logger.warning(f"Failed to download {key}: {e}")
        return "failed"


def download_images(metadata_path: Path, output_dir: Path, prefix: str, s3_client=None) -> dict:
    """Download every image listed in a metadata csv. Returns how many were downloaded/skipped/failed."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found, run `dvc pull` first")

    s3_client = s3_client or make_s3_client()
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = pd.read_csv(metadata_path)["filename"]

    # 8 downloads at the same time, like in the notebook
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda f: download_file(s3_client, prefix + f, output_dir / f), filenames)
        )

    counts = {status: results.count(status) for status in ("downloaded", "skipped", "failed")}
    logger.info(f"{output_dir}: {counts}")
    return counts


@app.command()
def main():
    """Download the training images and the test images."""
    download_images(METADATA_PATH, RAW_IMAGES_DIR, prefix="images/raw/")
    download_images(TEST_METADATA_PATH, TEST_DATA_DIR, prefix="test/")
    logger.success("Images downloaded")


if __name__ == "__main__":
    app()
