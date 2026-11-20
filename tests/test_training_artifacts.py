#testing_training Artefact
import json
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader, TensorDataset

import vial_classifier.modeling.train as train_module
from vial_classifier.config import IMAGE_SIZE
from vial_classifier.features import compute_baseline_statistics
from vial_classifier.modeling.model import SimpleCNN


def test_save_generated_training_artifacts_and_log_pyfunc(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "artifacts" / "vial_cnn_model.pth"
    results_path = tmp_path / "artifacts" / "model_results.json"
    baseline_path = tmp_path / "artifacts" / "drift_baseline_stats.json"
    monkeypatch.setattr(train_module, "PYTORCH_MODEL_PATH", model_path)
    monkeypatch.setattr(train_module, "MODEL_RESULTS_PATH", results_path)
    monkeypatch.setattr(train_module, "DRIFT_BASELINE_STATS_PATH", baseline_path)

    log_param = Mock()
    log_metric = Mock()
    log_artifact = Mock()
    log_model = Mock()
    monkeypatch.setattr(train_module.mlflow, "log_param", log_param)
    monkeypatch.setattr(train_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(train_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(train_module.mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(train_module.mlflow, "active_run", lambda: None)

    captured_wrapper = {}

    class WrapperStub:
        def __init__(self, model, threshold):
            captured_wrapper["model"] = model
            captured_wrapper["threshold"] = threshold

    monkeypatch.setattr(train_module, "SimpleLoggingWrapper", WrapperStub)
    torch.manual_seed(42)
    images = torch.rand((4, 3, *IMAGE_SIZE))
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = SimpleCNN().eval()
    baseline = compute_baseline_statistics(model, loader, device=torch.device("cpu"))
    threshold = train_module.tune_threshold(model, loader, device="cpu")
    results = train_module.evaluate_cnn_model(
        model, loader, device="cpu", threshold=threshold
    )

    train_module.save_training_artifacts(model, results, baseline, threshold)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert checkpoint["architecture"] == "SimpleCNN"
    assert checkpoint["input_shape"] == (3, *IMAGE_SIZE)
    assert checkpoint["output_shape"] == (1,)
    assert checkpoint["threshold"] == threshold
    assert checkpoint["num_classes"] == train_module.NUM_CLASSES
    assert set(checkpoint["model_state_dict"]) == set(model.state_dict())
    assert json.loads(results_path.read_text(encoding="utf-8")) == results
    assert json.loads(baseline_path.read_text(encoding="utf-8")) == baseline

    log_param.assert_called_once_with("classification_threshold", threshold)
    logged_metrics = {
        call.args[0]: call.args[1] for call in log_metric.call_args_list
    }
    assert logged_metrics["baseline_image_mean_r"] == (
        baseline["image_statistics"]["mean_r"]
    )
    assert logged_metrics["baseline_class_1_count"] == (
        baseline["class_distribution"]["class_1_count"]
    )
    assert logged_metrics["baseline_confidence_std"] == (
        baseline["confidence_statistics"]["std"]
    )
    assert [call.args for call in log_artifact.call_args_list] == [
        (str(baseline_path), "drift_detection"),
        (str(model_path),),
        (str(results_path),),
    ]
    log_model.assert_called_once()
    assert log_model.call_args.kwargs["artifact_path"] == (
        train_module.MLFLOW_MODEL_ARTIFACT_PATH
    )
    assert isinstance(log_model.call_args.kwargs["python_model"], WrapperStub)
    assert captured_wrapper == {"model": model, "threshold": threshold}

