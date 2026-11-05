from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch

from vial_classifier import dataset
from vial_classifier.features import define_transforms


def _write_jpeg(path):
    Image.new("RGB", (32, 16), color=(120, 60, 30)).save(path, format="JPEG")


def _write_metadata(path, rows):
    pd.DataFrame(rows, columns=["filename", "class"]).to_csv(path, index=False)


def _fake_items(n_class_0=20, n_class_1=80):
    """(path, label) pairs for the split tests, which never open the files."""
    return [(Path(f"clean_{i}.jpg"), 0) for i in range(n_class_0)] + [
        (Path(f"defect_{i}.jpg"), 1) for i in range(n_class_1)
    ]


def test_load_image_dataset_skips_missing_files(tmp_path):
    _write_jpeg(tmp_path / "a.jpg")
    _write_jpeg(tmp_path / "b.jpg")
    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(metadata_path, [("a.jpg", 0), ("missing.jpg", 0), ("b.jpg", 1)])

    assert dataset.load_image_dataset(metadata_path, tmp_path) == [
        (tmp_path / "a.jpg", 0),
        (tmp_path / "b.jpg", 1),
    ]


def test_load_image_dataset_requires_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.load_image_dataset(tmp_path / "metadata.csv", tmp_path)


def test_split_sizes_and_class_balance():
    train, val, test = dataset.split_dataset(_fake_items())

    counts = [[sum(y == c for _, y in part) for c in (0, 1)] for part in (train, val, test)]
    assert counts == [[14, 56], [3, 12], [3, 12]]


def test_split_keeps_every_item_exactly_once():
    items = _fake_items()
    train, val, test = dataset.split_dataset(items)

    assert sorted(train + val + test) == sorted(items)


def test_same_seed_gives_same_split():
    first = dataset.split_dataset(_fake_items(), random_seed=7)
    second = dataset.split_dataset(_fake_items(), random_seed=7)
    other = dataset.split_dataset(_fake_items(), random_seed=8)

    assert first == second
    assert first != other


def test_glass_vial_dataset_returns_tensor_and_label(tmp_path):
    path = tmp_path / "a.jpg"
    _write_jpeg(path)
    _, val_transforms = define_transforms()

    image, label = dataset.GlassVialDataset([path], [1], val_transforms)[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == 1


def test_create_dataloaders_batches_images(tmp_path):
    items = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        _write_jpeg(path)
        items.append((path, i % 2))
    train_transforms, val_transforms = define_transforms()

    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        items, items[:2], items[:1], train_transforms, val_transforms, batch_size=2
    )

    images, labels = next(iter(train_loader))
    assert images.shape == (2, 3, 64, 64)
    assert labels.shape == (2,)
    assert (len(train_loader), len(val_loader), len(test_loader)) == (2, 1, 1)


class _FakeS3Client:
    """Writes a file for every key except those listed in fail_keys."""

    def __init__(self, fail_keys=()):
        self.fail_keys = set(fail_keys)
        self.requested = []

    def download_file(self, bucket, key, output_path):
        self.requested.append(key)
        if key in self.fail_keys:
            raise OSError("simulated download failure")
        Path(output_path).write_bytes(b"image")


def test_download_skips_existing_files_and_uses_prefix(tmp_path, monkeypatch):
    client = _FakeS3Client()
    monkeypatch.setattr(dataset, "get_s3_client", lambda: client)
    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(metadata_path, [("existing.jpg", 0), ("new.jpg", 1)])
    output_dir = tmp_path / "images"
    output_dir.mkdir()
    (output_dir / "existing.jpg").write_bytes(b"old")

    dataset.download_files_from_metadata(metadata_path, output_dir, "prefix/")

    assert client.requested == ["prefix/new.jpg"]
    assert (output_dir / "new.jpg").exists()
    assert (output_dir / "existing.jpg").read_bytes() == b"old"


def test_download_failure_raises(tmp_path, monkeypatch):
    client = _FakeS3Client(fail_keys=["bad.jpg"])
    monkeypatch.setattr(dataset, "get_s3_client", lambda: client)
    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(metadata_path, [("good.jpg", 0), ("bad.jpg", 1)])

    with pytest.raises(RuntimeError, match="1 images failed"):
        dataset.download_files_from_metadata(metadata_path, tmp_path / "images")
