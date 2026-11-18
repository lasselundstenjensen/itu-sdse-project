#It's a small helper, so we don't really have to repeat two monkypatch lines after every test.
# *the tests always run on the CPU, so they behave the same on every laptop an din GitHub Actions

import mlflow
import torch
import torch.nn as nn

from reestructured.evaluate import evaluate_cnn_model, tune_threshold


class FakeModel(nn.Module):
    """Returns its input as the logits, so we control the predictions."""

    def forward(self, x):
        return x


def make_loader():
    # the "images" act as logits: sigmoid gives roughly 0.05, 0.12, 0.88, 0.95
    images = torch.tensor([[-3.0], [-2.0], [2.0], [3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    return [(images, labels)]


def silence_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_param", lambda *args, **kwargs: None)


def test_tune_threshold_finds_separating_threshold(monkeypatch):
    silence_mlflow(monkeypatch)

    best = tune_threshold(
        FakeModel(), make_loader(), device=torch.device("cpu"), threshold=0.5
    )

    # must sit between the highest negative (0.119) and the lowest positive (0.881)
    assert 0.119 < best < 0.881


def test_evaluate_cnn_model_on_perfect_predictions(monkeypatch):
    silence_mlflow(monkeypatch)

    results = evaluate_cnn_model(
        FakeModel(), make_loader(), device=torch.device("cpu"), threshold=0.5
    )

    assert results["accuracy"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["roc_auc"] == 1.0
    assert results["confusion_matrix"] == [[2, 0], [0, 2]]