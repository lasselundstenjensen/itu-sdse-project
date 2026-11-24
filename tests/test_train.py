import torch
from torch.utils.data import DataLoader, Dataset

from vial_classifier.modeling.model import SimpleCNN
from vial_classifier.modeling.train import evaluate_cnn_model, train_cnn_model, tune_threshold

CPU = torch.device("cpu")


class FakeVialDataset(Dataset):
    """16 random 64x64 'images' with the same 1:3 class balance as the real data."""

    def __init__(self, n=16):
        generator = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 64, 64, generator=generator)
        self.labels = [0, 1, 1, 1] * (n // 4)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_loader():
    return DataLoader(FakeVialDataset(), batch_size=8)


def test_train_runs_one_epoch_and_records_history():
    model, history = train_cnn_model(make_loader(), make_loader(), num_epochs=1, device=CPU)
    assert isinstance(model, SimpleCNN)
    assert len(history["train_loss"]) == 1
    assert len(history["val_f1_macro"]) == 1


def test_tune_threshold_returns_threshold_between_zero_and_one():
    threshold = tune_threshold(SimpleCNN(), make_loader(), device=CPU)
    assert 0.0 < threshold < 1.0


def test_evaluate_returns_all_metrics():
    results = evaluate_cnn_model(SimpleCNN(), make_loader(), device=CPU)
    for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc", "confusion_matrix"]:
        assert metric in results