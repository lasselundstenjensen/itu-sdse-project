from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path
from typing import Callable, Sequence, TypedDict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from ccds.modeling.model import SimpleCNN


class TrainingHistory(TypedDict):
    train_loss: list[float]
    val_loss: list[float]
    train_accuracy: list[float]
    val_accuracy: list[float]
    train_f1: list[float]
    val_f1: list[float]
    val_f1_macro: list[float]
    best_val_f1_macro: float
    best_model_state: dict[str, torch.Tensor] | None


DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_THRESHOLD = 0.5
IMAGE_SIZE = (64, 64)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
METADATA_PATH = Path("data/images/metadata.csv")
IMAGE_DIR = Path("data/images/raw")
CHECKPOINT_PATH = Path("artifacts/vial_cnn_model.pth")

ImageRecord = tuple[Path, int]
ImageTransform = Callable[[Image.Image], torch.Tensor]


class VialImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, records: Sequence[ImageRecord], transform: ImageTransform | None
    ) -> None:
        self.records = list(records)
        self.labels = [label for _, label in self.records]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_image_records(metadata_path: Path, image_dir: Path) -> list[ImageRecord]:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. Pull the DVC metadata and "
            "download the image files as in notebooks/01_training.ipynb first."
        )

    metadata = pd.read_csv(metadata_path)
    missing_columns = {"filename", "class"} - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata file {metadata_path} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    records: list[ImageRecord] = []
    missing_files: list[str] = []
    for filename_value, label_value in metadata[
        ["filename", "class"]
    ].itertuples(index=False, name=None):
        filename = str(filename_value)
        label = int(label_value)
        image_path = image_dir / filename
        if image_path.is_file():
            records.append((image_path, label))
        else:
            missing_files.append(filename)

    if missing_files:
        warnings.warn(
            f"Skipping {len(missing_files)} metadata entries with missing images "
            f"under {image_dir}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not records:
        raise ValueError(f"No images from {metadata_path} were found under {image_dir}.")
    if any(label not in (0, 1) for _, label in records):
        raise ValueError("The metadata class column must contain only 0 and 1.")
    return records


def split_records(
    records: Sequence[ImageRecord],
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    np.random.seed(seed)
    class_records = {
        label: [record for record in records if record[1] == label]
        for label in (0, 1)
    }
    if any(not items for items in class_records.values()):
        raise ValueError("Training data must contain at least one image of each class.")

    def split_class(
        items: list[ImageRecord],
    ) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
        np.random.shuffle(items)
        train_size = int(len(items) * train_ratio)
        validation_size = int(len(items) * validation_ratio)
        return (
            items[:train_size],
            items[train_size : train_size + validation_size],
            items[train_size + validation_size :],
        )

    train_0, validation_0, test_0 = split_class(class_records[0])
    train_1, validation_1, test_1 = split_class(class_records[1])
    train_records = train_0 + train_1
    validation_records = validation_0 + validation_1
    test_records = test_0 + test_1
    np.random.shuffle(train_records)
    np.random.shuffle(validation_records)
    np.random.shuffle(test_records)

    if not train_records or not validation_records or not test_records:
        raise ValueError(
            "The stratified 70/15/15 split produced an empty partition; "
            "provide more labeled images."
        )
    return train_records, validation_records, test_records


def create_transforms() -> tuple[ImageTransform, ImageTransform]:
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return train_transform, validation_transform


def train_cnn_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: torch.device | str = "cpu",
    random_seed: int = DEFAULT_SEED,
) -> tuple[SimpleCNN, TrainingHistory]:
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("Training and validation data loaders must not be empty.")

    seed_everything(random_seed)
    model = SimpleCNN().to(device)
    train_labels = train_loader.dataset.labels
    positive_count = int(sum(train_labels))
    negative_count = len(train_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both classes.")

    pos_weight = torch.tensor(
        [negative_count / positive_count], dtype=torch.float32, device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history: TrainingHistory = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
        "val_f1_macro": [],
        "best_val_f1_macro": -1.0,
        "best_model_state": None,
    }
    epochs_without_improvement = 0

    for _epoch in range(num_epochs):
        model.train()
        train_losses: list[float] = []
        train_predictions: list[float] = []
        train_targets: list[float] = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_predictions.extend(
                (torch.sigmoid(logits) >= DEFAULT_THRESHOLD)
                .float()
                .view(-1)
                .cpu()
                .tolist()
            )
            train_targets.extend(labels.view(-1).cpu().tolist())

        model.eval()
        validation_losses: list[float] = []
        validation_predictions: list[float] = []
        validation_targets: list[float] = []
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)
                logits = model(images)
                validation_losses.append(criterion(logits, labels).item())
                validation_predictions.extend(
                    (torch.sigmoid(logits) >= DEFAULT_THRESHOLD)
                    .float()
                    .view(-1)
                    .cpu()
                    .tolist()
                )
                validation_targets.extend(labels.view(-1).cpu().tolist())

        train_f1 = f1_score(train_targets, train_predictions, zero_division=0)
        validation_f1 = f1_score(
            validation_targets, validation_predictions, zero_division=0
        )
        validation_f1_macro = f1_score(
            validation_targets,
            validation_predictions,
            average="macro",
            zero_division=0,
        )

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(validation_losses)))
        history["train_accuracy"].append(
            float(np.mean(np.equal(train_targets, train_predictions)))
        )
        history["val_accuracy"].append(
            float(np.mean(np.equal(validation_targets, validation_predictions)))
        )
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(float(validation_f1))
        history["val_f1_macro"].append(float(validation_f1_macro))

        scheduler.step(validation_f1_macro)
        if validation_f1_macro > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = validation_f1_macro
            history["best_model_state"] = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 2:
                break

    best_model_state = history["best_model_state"]
    if best_model_state is None:
        raise RuntimeError("Training did not produce a best model checkpoint.")
    model.load_state_dict(best_model_state)
    return model, history


def tune_threshold(
    model: SimpleCNN,
    validation_loader: DataLoader,
    device: torch.device | str,
) -> float:
    probabilities: list[float] = []
    labels: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in validation_loader:
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
            labels.extend(batch_labels.float().view(-1).cpu().tolist())

    if not labels:
        raise ValueError("Cannot tune a threshold on an empty validation set.")

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = 0.0
    for candidate in np.arange(0.025, 0.976, 0.025):
        predictions = (np.asarray(probabilities) >= candidate).astype(int)
        macro_f1 = f1_score(
            labels, predictions, average="macro", zero_division=0
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = round(float(candidate), 3)
    return best_threshold


def save_checkpoint(
    model: SimpleCNN, path: Path, threshold: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": SimpleCNN.__name__,
            "input_shape": (3, 64, 64),
            "output_shape": (1,),
            "threshold": threshold,
            "num_classes": 2,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[SimpleCNN, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint at {path}: expected a dictionary.")
    if checkpoint.get("architecture") != SimpleCNN.__name__:
        raise ValueError(
            f"Unsupported checkpoint architecture: {checkpoint.get('architecture')!r}"
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint at {path}: missing model_state_dict.")
    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the vial image CNN.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    records = load_image_records(args.metadata, args.images)
    train_records, validation_records, _test_records = split_records(
        records, seed=args.seed
    )
    train_transform, validation_transform = create_transforms()
    train_loader = DataLoader(
        VialImageDataset(train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        VialImageDataset(validation_records, validation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _history = train_cnn_model(
        train_loader,
        validation_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        random_seed=args.seed,
    )
    threshold = tune_threshold(model, validation_loader, device)
    save_checkpoint(model, args.output, threshold)
    print(f"Saved SimpleCNN checkpoint to {args.output}")


if __name__ == "__main__":
    main()
