"""Tests for src/features.py: metadata validation and the stratified split."""

import pandas as pd
import pytest

from src.features import load_image_dataset, split_dataset, validate_metadata


def fake_dataset(n_clean=20, n_defective=80):
    """A list of (filename, label) like the real one: 20% clean, 80% defective."""
    return [(f"clean_{i}.jpg", 0) for i in range(n_clean)] + [
        (f"defective_{i}.jpg", 1) for i in range(n_defective)
    ]


def make_metadata(tmp_path, filenames, labels):
    """Write a metadata csv and create empty image files for it."""
    for f in filenames:
        (tmp_path / f).touch()
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"filename": filenames, "class": labels}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- split


def test_split_sizes_are_70_15_15():
    train, val, test = split_dataset(fake_dataset())
    assert (len(train), len(val), len(test)) == (70, 15, 15)


def test_split_is_stratified():
    train, val, test = split_dataset(fake_dataset())
    # every set keeps the 20% clean / 80% defective balance
    assert [label for _, label in train].count(0) == 14
    assert [label for _, label in val].count(0) == 3
    assert [label for _, label in test].count(0) == 3


def test_split_keeps_every_image_exactly_once():
    data = fake_dataset()
    train, val, test = split_dataset(data)
    assert sorted(train + val + test) == sorted(data)


def test_split_is_reproducible():
    assert split_dataset(fake_dataset()) == split_dataset(fake_dataset())


def test_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        split_dataset(fake_dataset(), train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)


# ---------------------------------------------------------------- validation


def test_valid_metadata_loads(tmp_path):
    path = make_metadata(tmp_path, ["a.jpg", "b.jpg"], [0, 1])
    assert load_image_dataset(path, tmp_path) == [("a.jpg", 0), ("b.jpg", 1)]


@pytest.mark.parametrize(
    "metadata, problem",
    [
        (pd.DataFrame({"filename": ["a.jpg"]}), "missing columns"),
        (pd.DataFrame({"filename": ["a.jpg"], "class": [2]}), "Unexpected labels"),
        (pd.DataFrame({"filename": ["a.jpg", "a.jpg"], "class": [0, 0]}), "Duplicate"),
        (pd.DataFrame({"filename": ["not_there.jpg"], "class": [0]}), "missing in"),
    ],
)
def test_bad_metadata_is_rejected(tmp_path, metadata, problem):
    (tmp_path / "a.jpg").touch()
    with pytest.raises(ValueError, match=problem):
        validate_metadata(metadata, tmp_path)


# ---------------------------------------------------------------- transforms + Dataset


def test_eval_transform_gives_a_3x64x64_tensor():
    from PIL import Image

    from src.features import get_transforms

    _, eval_transforms = get_transforms()
    image = Image.new("RGB", (1600, 768))  # same size as the real photos
    assert tuple(eval_transforms(image).shape) == (3, 64, 64)


def test_dataset_returns_image_and_label(tmp_path):
    from PIL import Image

    from src.features import GlassVialDataset, get_transforms

    Image.new("RGB", (1600, 768)).save(tmp_path / "a.jpg")
    Image.new("L", (1600, 768)).save(tmp_path / "b.jpg")  # greyscale, must become RGB
    split_csv = tmp_path / "train.csv"
    pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "label": [0, 1]}).to_csv(split_csv, index=False)

    _, eval_transforms = get_transforms()
    dataset = GlassVialDataset(split_csv, tmp_path, transform=eval_transforms)

    assert len(dataset) == 2
    image, label = dataset[1]
    assert tuple(image.shape) == (3, 64, 64)
    assert label == 1
