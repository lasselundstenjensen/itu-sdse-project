"""Tests for src/dataset.py, with a fake S3 client so no internet is needed."""

from pathlib import Path

import pandas as pd
import pytest

from src.dataset import download_file, download_images


class FakeS3:
    """Pretends to be the bucket: 'downloading' just writes a small file."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def download_file(self, bucket, key, path):
        self.calls.append(key)
        if key in self.missing:
            raise Exception("404 not found")
        Path(path).write_text("fake image")


def test_download_file_skips_existing(tmp_path):
    existing = tmp_path / "a.jpg"
    existing.write_text("already here")
    s3 = FakeS3()
    assert download_file(s3, "images/raw/a.jpg", existing) == "skipped"
    assert s3.calls == []  # nothing was downloaded


def test_download_images_counts(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"filename": ["a.jpg", "b.jpg", "c.jpg"], "class": [0, 1, 1]}).to_csv(
        metadata, index=False
    )
    out = tmp_path / "raw"
    out.mkdir()
    (out / "a.jpg").write_text("already here")

    counts = download_images(metadata, out, "images/raw/", s3_client=FakeS3(missing={"images/raw/c.jpg"}))

    assert counts == {"downloaded": 1, "skipped": 1, "failed": 1}
    assert (out / "b.jpg").exists()


def test_download_images_without_metadata_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        download_images(tmp_path / "missing.csv", tmp_path, "images/raw/", s3_client=FakeS3())
