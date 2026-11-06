import mlflow
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vial_classifier.modeling.evaluate import (
    compute_baseline_statistics,
    evaluate_cnn_model,
)
from vial_classifier.modeling.model import SimpleCNN


class FakeVialDataset(Dataset):
    """Tiny random dataset with both classes."""

    def __init__(self, n=8):
        self.labels = [i % 2 for i in range(n)]
        self.images = torch.randn(n, 3, 64, 64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_loader():
    return DataLoader(FakeVialDataset(), batch_size=4, shuffle=False)


CPU = torch.device("cpu")


def test_evaluate_cnn_model_returns_metrics(monkeypatch):
    logged = []
    monkeypatch.setattr(mlflow, "log_metric", lambda name, value, **kw: logged.append(name))

    results = evaluate_cnn_model(SimpleCNN(), make_loader(), device=CPU)

    for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        assert 0.0 <= results[key] <= 1.0
    assert len(results["confusion_matrix"]) == 2
    assert "test_f1_score" in logged
    assert "test_roc_auc" in logged


def test_compute_baseline_statistics_structure():
    stats = compute_baseline_statistics(SimpleCNN(), make_loader(), device=CPU)

    assert stats["class_distribution"] == {
        "class_0_count": 4,
        "class_1_count": 4,
        "total": 8,
    }
    assert stats["prediction_distribution"]["total"] == 8
    assert stats["embedding_statistics"]["shape"] == 128  # fc1 output size
    assert len(stats["embedding_statistics"]["mean"]) == 128
    assert 0.0 <= stats["confidence_statistics"]["mean"] <= 1.0


def test_compute_baseline_statistics_rejects_unknown_layer():
    with pytest.raises(ValueError):
        compute_baseline_statistics(
            SimpleCNN(), make_loader(), device=CPU, feature_layer="does_not_exist"
        )
