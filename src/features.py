"""Prepare the data for training (data validation + preparation step).

1. Load the metadata and check it is valid (fail early on bad data).
2. Split it into train / validation / test, stratified by class, with a
   fixed seed: exactly the same split as the original notebook.
3. Save the three sets as csv files in data/processed/, which is what
   the training step reads.

It also holds the image transforms and the PyTorch Dataset that turn those
csv files into tensors, so training and serving prepare images the same way.

Run it with:  uv run python -m src.features
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import typer

from src.config import (
    IMAGE_SIZE,
    METADATA_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_IMAGES_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    VAL_RATIO,
)

app = typer.Typer()

REQUIRED_COLUMNS = ["filename", "class"]
VALID_LABELS = {0, 1}  # 0 = clean, 1 = defective


def validate_metadata(metadata: pd.DataFrame, images_dir: Path) -> None:
    """Raise ValueError if the metadata is not usable for training."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {missing_columns}")

    bad_labels = set(metadata["class"].unique()) - VALID_LABELS
    if bad_labels:
        raise ValueError(f"Unexpected labels {bad_labels}, only 0 and 1 are allowed")

    duplicates = metadata["filename"][metadata["filename"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate filenames in metadata: {duplicates[:5]}")

    missing_images = [f for f in metadata["filename"] if not (images_dir / f).exists()]
    if missing_images:
        raise ValueError(
            f"{len(missing_images)} images are missing in {images_dir}, run `python -m src.dataset`"
        )


def load_image_dataset(metadata_path: Path, images_dir: Path) -> list[tuple[str, int]]:
    """Read and validate the metadata. Returns a list of (filename, label)."""
    metadata = pd.read_csv(metadata_path)
    validate_metadata(metadata, images_dir)
    logger.info(f"Loaded {len(metadata)} images, class counts {metadata['class'].value_counts().to_dict()}")
    return list(zip(metadata["filename"], metadata["class"].astype(int)))


def split_dataset(
    image_dataset: list[tuple[str, int]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_STATE,
):
    """Stratified split: each class is shuffled and split on its own, so every set
    keeps the same clean/defective balance. Same algorithm and seed as the notebook."""
    if abs(train_ratio + val_ratio + test_ratio - 1) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1")

    np.random.seed(random_seed)

    class_0 = [item for item in image_dataset if item[1] == 0]
    class_1 = [item for item in image_dataset if item[1] == 1]
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        return data[:train_size], data[train_size : train_size + val_size], data[train_size + val_size :]

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train, val, test = train_0 + train_1, val_0 + val_1, test_0 + test_1
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)
    return train, val, test


def save_splits(train, val, test, output_dir: Path) -> None:
    """Write train.csv, val.csv and test.csv with the columns filename,label."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame(data, columns=["filename", "label"]).to_csv(output_dir / f"{name}.csv", index=False)
        logger.info(f"{name}: {len(data)} images")


def get_transforms():
    """Image transforms (same as the notebook).

    train: resize + light augmentation (flip, rotation, colour jitter) + normalise.
    eval:  resize + normalise only. Validation, test AND the API must use this one,
           otherwise the model sees different inputs in production (training/serving skew).
    """
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])
    return train_transforms, eval_transforms


class GlassVialDataset(Dataset):
    """Reads one of the split csv files (filename,label) and gives (image tensor, label)."""

    def __init__(self, split_csv: Path, images_dir: Path = RAW_IMAGES_DIR, transform=None):
        split = pd.read_csv(split_csv)
        self.image_paths = [images_dir / f for f in split["filename"]]
        self.labels = split["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


@app.command()
def main(
    metadata_path: Path = METADATA_PATH,
    images_dir: Path = RAW_IMAGES_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """Validate the metadata, split it and save the three sets."""
    image_dataset = load_image_dataset(metadata_path, images_dir)
    train, val, test = split_dataset(image_dataset)
    save_splits(train, val, test, output_dir)
    logger.success(f"Splits saved to {output_dir}")


if __name__ == "__main__":
    app()
