"""Tests for training, threshold tuning and evaluation.

MLflow is replaced by a mock, so these tests never touch a real tracking store.
"""

from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vial_classifier import config
from vial_classifier.modeling import evaluate, train
from vial_classifier.modeling.model import SimpleCNN

CPU = torch.device("cpu")

ignore_undefined_metrics = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)


@pytest.fixture(autouse=True)
def mlflow_mock(monkeypatch):
    mock = MagicMock()
    mock.active_run.return_value = None
    monkeypatch.setattr(train, "mlflow", mock)
    monkeypatch.setattr(evaluate, "mlflow", mock)
    return mock


class FakeVialDataset(Dataset):
    """Random images with a `.labels` attribute, like GlassVialDataset."""

    def __init__(self, labels):
        self.labels = labels
        generator = torch.Generator().manual_seed(0)
        self.images = torch.rand(len(labels), 3, 64, 64, generator=generator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _loaders(labels=(0, 1, 0, 1, 0, 1, 0, 1)):
    dataset = FakeVialDataset(list(labels))
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader


def _script_val_macro_f1(monkeypatch, values):
    """Make each epoch's validation macro F1 follow `values`."""
    remaining = iter(values)

    def fake_f1_score(y_true, y_pred, average="binary"):
        return next(remaining) if average == "macro" else 0.0

    monkeypatch.setattr(train, "f1_score", fake_f1_score)


class FixedProbabilityModel(torch.nn.Module):
    """Treats each input as the probability to predict, returned as a logit."""

    def forward(self, x):
        return torch.logit(x)


def _fixed_probability_loader(probabilities, labels):
    return [(torch.tensor(probabilities).view(-1, 1), torch.tensor(labels))]


# --- setup_mlflow_tracking ---------------------------------------------------


@pytest.fixture
def mlflow_state_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "mlflow-state"
    monkeypatch.setattr(config, "MLFLOW_STATE_DIR", state_dir)
    return state_dir


def test_setup_mlflow_creates_missing_experiment(mlflow_mock, mlflow_state_dir):
    mlflow_mock.get_experiment_by_name.return_value = None

    train.setup_mlflow_tracking("my_experiment")

    assert mlflow_state_dir.is_dir()
    mlflow_mock.set_tracking_uri.assert_called_once_with(config.MLFLOW_TRACKING_URI)
    mlflow_mock.create_experiment.assert_called_once_with(
        "my_experiment", artifact_location=config.MLFLOW_ARTIFACT_LOCATION
    )
    mlflow_mock.set_experiment.assert_called_once_with("my_experiment")


def test_setup_mlflow_reuses_existing_experiment(mlflow_mock, mlflow_state_dir):
    train.setup_mlflow_tracking("my_experiment")

    mlflow_mock.create_experiment.assert_not_called()
    mlflow_mock.set_experiment.assert_called_once_with("my_experiment")


def test_setup_mlflow_without_experiment_only_sets_uri(mlflow_mock, mlflow_state_dir):
    train.setup_mlflow_tracking(None)

    mlflow_mock.set_tracking_uri.assert_called_once_with(config.MLFLOW_TRACKING_URI)
    mlflow_mock.set_experiment.assert_not_called()


# --- train_cnn_model ---------------------------------------------------------


def test_training_returns_history_and_restores_best_weights(monkeypatch):
    # The first epoch is the best, so the second epoch's weights must be thrown away
    _script_val_macro_f1(monkeypatch, [0.9, 0.1])
    weights_before_restore = {}
    original_load_state_dict = SimpleCNN.load_state_dict

    def spy_load_state_dict(self, state_dict, *args, **kwargs):
        weights_before_restore.update({k: v.clone() for k, v in self.state_dict().items()})
        return original_load_state_dict(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(SimpleCNN, "load_state_dict", spy_load_state_dict)
    train_loader, val_loader = _loaders()

    model, history = train.train_cnn_model(train_loader, val_loader, num_epochs=2, device=CPU)

    for key in (
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_f1",
        "val_f1",
        "val_f1_macro",
    ):
        assert len(history[key]) == 2
    assert history["best_val_f1_macro"] == 0.9

    best = history["best_model_state"]
    assert all(torch.equal(value, best[key]) for key, value in model.state_dict().items())
    assert not all(torch.equal(value, best[key]) for key, value in weights_before_restore.items())


def test_early_stopping_after_two_epochs_without_improvement(monkeypatch):
    # Ties don't count as an improvement, and 0.9 is never reached
    _script_val_macro_f1(monkeypatch, [0.5, 0.5, 0.1, 0.9, 0.9])
    train_loader, val_loader = _loaders()

    _, history = train.train_cnn_model(train_loader, val_loader, num_epochs=5, device=CPU)

    assert len(history["val_loss"]) == 3
    assert history["best_val_f1_macro"] == 0.5


@ignore_undefined_metrics
def test_seed_at_start_makes_training_reproducible():
    train_loader, val_loader = _loaders()

    first, _ = train.train_cnn_model(train_loader, val_loader, num_epochs=1, device=CPU)
    # Earlier steps such as the image statistics consume random numbers before training
    torch.rand(100)
    np.random.rand(100)
    second, _ = train.train_cnn_model(train_loader, val_loader, num_epochs=1, device=CPU)

    second_weights = second.state_dict()
    assert all(torch.equal(value, second_weights[k]) for k, value in first.state_dict().items())


def test_loss_is_weighted_by_negative_to_positive_ratio(monkeypatch):
    _script_val_macro_f1(monkeypatch, [0.5])
    pos_weights = []
    original_loss = train.nn.BCEWithLogitsLoss

    def spy_loss(pos_weight):
        pos_weights.append(pos_weight.item())
        return original_loss(pos_weight=pos_weight)

    monkeypatch.setattr(train.nn, "BCEWithLogitsLoss", spy_loss)
    train_loader, val_loader = _loaders(labels=(0, 0, 1, 1, 1, 1, 1, 1))

    train.train_cnn_model(train_loader, val_loader, num_epochs=1, device=CPU)

    assert pos_weights == pytest.approx([2 / 6])


def test_no_metrics_logged_without_active_run(mlflow_mock, monkeypatch):
    _script_val_macro_f1(monkeypatch, [0.5])
    train_loader, val_loader = _loaders()

    train.train_cnn_model(train_loader, val_loader, num_epochs=1, device=CPU)

    mlflow_mock.log_metric.assert_not_called()


def test_epoch_that_stops_early_is_not_logged(mlflow_mock, monkeypatch):
    # Notebook quirk: `break` comes before the logging, so only two of three epochs are logged
    mlflow_mock.active_run.return_value = object()
    _script_val_macro_f1(monkeypatch, [0.5, 0.1, 0.1])
    train_loader, val_loader = _loaders()

    train.train_cnn_model(train_loader, val_loader, num_epochs=5, device=CPU)

    assert mlflow_mock.log_metric.call_args_list.count(call("val_f1_macro", 0.5, step=0)) == 1
    assert mlflow_mock.log_metric.call_args_list.count(call("val_f1_macro", 0.1, step=1)) == 1
    assert call("val_f1_macro", 0.1, step=2) not in mlflow_mock.log_metric.call_args_list


# --- tune_threshold ----------------------------------------------------------


def test_tune_threshold_picks_the_best_threshold_and_logs_it(mlflow_mock):
    loader = _fixed_probability_loader([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])

    threshold = train.tune_threshold(
        FixedProbabilityModel(), loader, device=CPU, threshold_range=[0.05, 0.5, 0.95]
    )

    assert threshold == 0.5
    mlflow_mock.log_param.assert_called_once_with("optimal_threshold", 0.5)
    mlflow_mock.log_metric.assert_called_once_with("optimal_threshold_macro_f1", 1.0)


def test_tune_threshold_default_grid_keeps_lowest_of_tied_thresholds():
    # Every threshold from 0.225 up to 0.8 separates the classes perfectly
    loader = _fixed_probability_loader([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])

    assert train.tune_threshold(FixedProbabilityModel(), loader, device=CPU) == 0.225


# --- evaluate_cnn_model ------------------------------------------------------


def test_evaluate_returns_all_metric_keys():
    loader = _fixed_probability_loader([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1])

    results = evaluate.evaluate_cnn_model(FixedProbabilityModel(), loader, device=CPU)

    assert set(results) == {
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    }


def test_evaluate_computes_expected_values():
    loader = _fixed_probability_loader([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1])

    results = evaluate.evaluate_cnn_model(
        FixedProbabilityModel(), loader, device=CPU, threshold=0.5
    )

    assert results["accuracy"] == 0.5
    assert results["precision"] == 0.5
    assert results["recall"] == 0.5
    assert results["f1_score"] == 0.5
    assert results["roc_auc"] == 0.75
    assert results["confusion_matrix"] == [[1, 1], [1, 1]]


def test_evaluate_uses_the_given_threshold():
    loader = _fixed_probability_loader([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1])

    results = evaluate.evaluate_cnn_model(
        FixedProbabilityModel(), loader, device=CPU, threshold=0.3
    )

    assert results["accuracy"] == 0.75
    assert results["confusion_matrix"] == [[1, 1], [0, 2]]


def test_evaluate_logs_the_metrics_deployment_relies_on(mlflow_mock):
    loader = _fixed_probability_loader([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1])

    results = evaluate.evaluate_cnn_model(
        FixedProbabilityModel(), loader, device=CPU, threshold=0.5
    )

    logged = {c.args[0]: c.args[1] for c in mlflow_mock.log_metric.call_args_list}
    assert logged == {
        "test_accuracy": results["accuracy"],
        "test_f1_score": results["f1_score"],
        "test_precision": results["precision"],
        "test_recall": results["recall"],
        "test_roc_auc": results["roc_auc"],
    }
