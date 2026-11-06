from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from vial_classifier import dataset
from vial_classifier.dataset import (
    _download_file_with_boto3,
    download_files_from_metadata,
    load_image_dataset,
)


class FakeS3:
    """Stands in for the boto3 client: 'downloads' by writing a tiny file. No internet needed."""

    def download_file(self, bucket, key, output_path):
        Path(output_path).write_bytes(b"fake image")


def make_metadata(path, filenames, classes):
    """Write a mini metadata.csv with only the columns our code reads."""
    pd.DataFrame({"filename": filenames, "class": classes}).to_csv(path, index=False)


def test_load_image_dataset_skips_missing_files(tmp_path):
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    Image.new("RGB", (16, 16)).save(image_dir / "a.jpg")
    metadata = tmp_path / "metadata.csv"
    make_metadata(metadata, ["a.jpg", "missing.jpg"], [0, 1])

    result = load_image_dataset(metadata_path=metadata, image_data_dir=image_dir)

    assert result == [(image_dir / "a.jpg", 0)]


def test_load_image_dataset_raises_without_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_dataset(metadata_path=tmp_path / "nope.csv", image_data_dir=tmp_path)


def test_download_skips_existing_file(tmp_path):
    existing = tmp_path / "a.jpg"
    existing.write_bytes(b"already here")

    assert _download_file_with_boto3(FakeS3(), "images/raw/a.jpg", existing) is False
    assert existing.read_bytes() == b"already here"  # not overwritten


def test_download_files_from_metadata_only_downloads_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dataset, "_setup_s3_client", lambda: FakeS3())
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / "a.jpg").write_bytes(b"already here")
    metadata = tmp_path / "metadata.csv"
    make_metadata(metadata, ["a.jpg", "b.jpg"], [0, 1])

    download_files_from_metadata(metadata, out_dir, prefix="images/raw/")

    assert (out_dir / "b.jpg").exists()
    assert "Downloaded 1 images, skipped 1 existing, failed 0" in capsys.readouterr().out
