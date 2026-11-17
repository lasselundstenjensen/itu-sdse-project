"""Stage 3 integration tests using temporary synthetic vial images.

Run from repository root: python -m pytest -q tests/test_integration.py
No DVC download, object-store access, or MLflow run is required.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from vial_classifier import dataset as dataset_module
from vial_classifier.config import IMAGE_SIZE
from vial_classifier.dataset import (
    GlassVialDataset,
    create_dataloaders,
    define_transforms,
    load_image_dataset,
    split_dataset,
)
from vial_classifier.features import calculate_image_statistics, compute_baseline_statistics
from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.train import (
    evaluate_cnn_model,
    train_cnn_model,
    tune_threshold,
)


@pytest.fixture
def synthetic_vials(tmp_path):
    """20 tiny RGB images, balanced between good and defective labels."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = ["filename,class"]
    for i in range(20):
        label = i % 2
        name = f"vial_{i:02d}.png"
        pixels = np.full((24, 24, 3), 60 if label == 0 else 180, dtype=np.uint8)
        pixels[:, :, i % 3] = (pixels[:, :, i % 3].astype(np.int16) + i) % 256
        Image.fromarray(pixels, mode="RGB").save(image_dir / name)
        rows.append(f"{name},{label}")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return image_dir, metadata


@pytest.fixture
def pipeline_loaders(synthetic_vials, monkeypatch):
    image_dir, metadata = synthetic_vials
    monkeypatch.setattr(dataset_module, "IMAGE_DATA_DIR", image_dir)
    monkeypatch.setattr(dataset_module, "METADATA_PATH", metadata)
    pairs = load_image_dataset()
    train, val, test = split_dataset(pairs, random_seed=42)
    loaders = create_dataloaders(train, val, test, batch_size=4)
    return pairs, (train, val, test), loaders


def test_metadata_to_dataset_to_dataloaders(pipeline_loaders):
    pairs, splits, loaders = pipeline_loaders
    assert len(pairs) == 20
    assert {label for _, label in pairs} == {0, 1}
    assert sum(map(len, splits)) == 20
    assert [len(split) for split in splits] == [14, 2, 4]
    assert all({label for _, label in split} == {0, 1} for split in splits)
    assert len({str(path) for split in splits for path, _ in split}) == 20
    for loader, split in zip(loaders, splits):
        assert len(loader.dataset) == len(split)
        images, labels = next(iter(loader))
        assert images.ndim == 4
        assert tuple(images.shape[1:]) == (3, *IMAGE_SIZE)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert set(labels.tolist()) <= {0, 1}
        assert torch.isfinite(images).all()


def test_transforms_and_dataset_handle_non_rgb_image(tmp_path):
    image_path = tmp_path / "grayscale.png"
    Image.new("L", (20, 20), color=120).save(image_path)
    train_transform, val_transform = define_transforms()
    for transform in (train_transform, val_transform):
        dataset = GlassVialDataset([image_path], [1], transform)
        image, label = dataset[0]
        assert image.shape == (3, *IMAGE_SIZE)
        assert label == 1
        assert torch.isfinite(image).all()


def test_image_statistics_and_model_forward(pipeline_loaders):
    _, _, (train_loader, _, _) = pipeline_loaders
    stats = calculate_image_statistics(train_loader)
    assert stats["total_samples"] == len(train_loader.dataset)
    assert len(stats["mean"]) == len(stats["std"]) == 3
    assert np.isfinite(stats["mean"]).all()
    assert np.isfinite(stats["std"]).all()
    images, _ = next(iter(train_loader))
    model = SimpleCNN().eval()
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (images.shape[0], 1)
    assert torch.isfinite(logits).all()


def test_one_epoch_training_threshold_and_evaluation(pipeline_loaders):
    _, _, (train_loader, val_loader, test_loader) = pipeline_loaders
    model, history = train_cnn_model(
        train_loader, val_loader, num_epochs=1, device="cpu", random_seed=42
    )
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert np.isfinite(history["train_loss"]).all()
    threshold = tune_threshold(model, val_loader, device="cpu")
    assert 0 < threshold < 1
    metrics = evaluate_cnn_model(model, test_loader, device="cpu", threshold=threshold)
    assert set(("accuracy", "f1_score", "precision", "recall", "roc_auc", "confusion_matrix")) <= set(metrics)
    assert all(0 <= metrics[key] <= 1 for key in ("accuracy", "f1_score", "precision", "recall"))
    assert len(metrics["confusion_matrix"]) == 2
    assert sum(map(sum, metrics["confusion_matrix"])) == len(test_loader.dataset)
    assert metrics["roc_auc"] is None or 0 <= metrics["roc_auc"] <= 1


def test_drift_baseline_structure(pipeline_loaders):
    _, _, (_, val_loader, _) = pipeline_loaders
    baseline = compute_baseline_statistics(SimpleCNN(), val_loader, device=torch.device("cpu"))
    assert baseline["class_distribution"]["total"] == len(val_loader.dataset)
    assert baseline["prediction_distribution"]["total"] == len(val_loader.dataset)
    assert len(baseline["embedding_statistics"]["mean"]) == 128
    assert len(baseline["embedding_statistics"]["std"]) == 128
    assert 0 <= baseline["confidence_statistics"]["min"] <= baseline["confidence_statistics"]["max"] <= 1
    json.dumps(baseline)  # Baseline can be serialized to the expected JSON format.
