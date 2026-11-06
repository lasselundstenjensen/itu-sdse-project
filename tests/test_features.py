import torch
from PIL import Image

from vial_classifier.features import (
    GlassVialDataset,
    calculate_image_statistics,
    create_dataloaders,
    define_transforms,
    split_dataset,
)


def fake_dataset(n_clean=20, n_defective=80):
    """(path, label) pairs like load_image_dataset returns. Paths are never opened by split_dataset."""
    return [(f"clean_{i}.jpg", 0) for i in range(n_clean)] + [
        (f"defective_{i}.jpg", 1) for i in range(n_defective)
    ]


def make_images(folder, n):
    """Write n small RGB jpgs with the real aspect ratio and return (path, label) pairs."""
    pairs = []
    for i in range(n):
        path = folder / f"img_{i}.jpg"
        Image.new("RGB", (160, 77), color=(i * 20, 100, 200)).save(path)
        pairs.append((path, i % 2))
    return pairs


def test_split_sizes_and_no_overlap():
    train, val, test = split_dataset(fake_dataset())

    # 20 clean -> 14/3/3, 80 defective -> 56/12/12
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    paths = [p for p, _ in train + val + test]
    assert len(paths) == len(set(paths)) == 100


def test_split_is_stratified():
    train, val, test = split_dataset(fake_dataset())

    for split in (train, val, test):
        labels = [label for _, label in split]
        assert 0 in labels and 1 in labels


def test_split_is_deterministic():
    assert split_dataset(fake_dataset()) == split_dataset(fake_dataset())


def test_val_transform_resizes_to_64x64():
    _, val_transforms = define_transforms()
    tensor = val_transforms(Image.new("RGB", (1600, 768)))

    assert tensor.shape == (3, 64, 64)


def test_dataset_returns_tensor_and_label(tmp_path):
    pairs = make_images(tmp_path, 2)
    _, val_transforms = define_transforms()
    ds = GlassVialDataset([p for p, _ in pairs], [label for _, label in pairs], val_transforms)

    image, label = ds[1]

    assert len(ds) == 2
    assert isinstance(image, torch.Tensor) and image.shape == (3, 64, 64)
    assert label == 1


def test_dataloaders_and_image_statistics(tmp_path):
    pairs = make_images(tmp_path, 6)
    train_loader, val_loader, test_loader = create_dataloaders(pairs[:4], pairs[4:5], pairs[5:], batch_size=2)

    assert len(train_loader) == 2  # 4 images / batch size 2
    stats = calculate_image_statistics(val_loader)
    assert stats["total_samples"] == 1
    assert len(stats["mean"]) == 3 and len(stats["std"]) == 3