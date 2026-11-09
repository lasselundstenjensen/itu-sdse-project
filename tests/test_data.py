import pandas as pd
import pytest

from vial_classifier import config
from vial_classifier.data import load_image_dataset, s3_key_from_url, split_dataset


def test_s3_key_strips_endpoint_and_bucket():
    url = f"{config.S3_ENDPOINT}/{config.S3_BUCKET}/images/metadata.csv"
    assert s3_key_from_url(url) == "images/metadata.csv"


def test_load_image_dataset_requires_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_dataset(metadata_path=tmp_path / "missing.csv", image_dir=tmp_path)


def test_load_image_dataset_skips_missing_images(tmp_path):
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"x")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "class": [0, 1]}).to_csv(metadata, index=False)

    samples = load_image_dataset(metadata_path=metadata, image_dir=image_dir)
    assert samples == [(image_dir / "a.jpg", 0)]


def _samples(n_per_class: int):
    clean = [(f"c{i}.jpg", 0) for i in range(n_per_class)]
    defective = [(f"d{i}.jpg", 1) for i in range(n_per_class)]
    return clean + defective


def test_split_is_stratified_and_covers_everything():
    samples = _samples(100)
    train, val, test = split_dataset(samples)

    assert len(train) + len(val) + len(test) == len(samples)
    assert len(train) == 140 and len(val) == 30 and len(test) == 30
    for part in (train, val, test):
        labels = [label for _, label in part]
        assert labels.count(0) == labels.count(1)


def test_split_is_reproducible_for_same_seed():
    samples = _samples(50)
    assert split_dataset(samples, random_seed=7) == split_dataset(samples, random_seed=7)
