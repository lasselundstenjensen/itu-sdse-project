import mlflow
import torch
from torch.utils.data import DataLoader, Dataset

from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.train import train_cnn_model, tune_threshold


class FakeVialDataset(Dataset):
    """Tiny random dataset with both classes and a `labels` attribute."""

    def __init__(self, n=8):
        self.labels = [i % 2 for i in range(n)]
        self.images = torch.randn(n, 3, 64, 64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_loader():
    return DataLoader(FakeVialDataset(), batch_size=4, shuffle=False)


def silence_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kwargs: None)


def test_train_cnn_model_returns_model_and_history(monkeypatch):
    silence_mlflow(monkeypatch)
    model, history = train_cnn_model(
        make_loader(), make_loader(), num_epochs=1, device=torch.device("cpu")
    )
    assert isinstance(model, SimpleCNN)
    assert len(history["train_loss"]) == 1
    assert len(history["val_f1_macro"]) == 1


def test_tune_threshold_returns_value_from_range(monkeypatch):
    silence_mlflow(monkeypatch)
    grid = [0.25, 0.5, 0.75]
    best = tune_threshold(
        SimpleCNN(), make_loader(), device=torch.device("cpu"), threshold_range=grid
    )
    assert best in grid or best == 0.5  # 0.5 is the fallback default