import json

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from vial_classifier import config
from vial_classifier.modeling.export import export_model


class DummyModel(mlflow.pyfunc.PythonModel):
    """Tiny stand-in for the real wrapper: export only copies files, it never runs the model."""

    def predict(self, context, model_input):
        return [0]


@pytest.fixture
def mlflow_tmp(tmp_path):
    """A fresh, empty MLflow store inside tmp_path, so tests never touch the real mlflow-state/."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.create_experiment(
        config.EXPERIMENT_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    yield tmp_path
    mlflow.set_tracking_uri(None)


def log_dummy_run():
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(config.MODEL_ARTIFACT_PATH, python_model=DummyModel())
    return run.info.run_id


def test_exports_production_model_with_version(mlflow_tmp):
    run_id = log_dummy_run()
    mv = mlflow.register_model(f"runs:/{run_id}/{config.MODEL_ARTIFACT_PATH}", config.MODEL_NAME)
    MlflowClient().set_registered_model_alias(config.MODEL_NAME, "Production", mv.version)

    out = export_model(export_dir=mlflow_tmp / "models" / "model")

    assert (out / "MLmodel").exists()
    version_info = json.loads((out / "version.json").read_text())
    assert version_info["model_version"] == "1"
    assert version_info["trained_at"] is not None


def test_falls_back_to_latest_run_when_nothing_registered(mlflow_tmp):
    log_dummy_run()

    out = export_model(export_dir=mlflow_tmp / "models" / "model")

    assert (out / "MLmodel").exists()
    assert json.loads((out / "version.json").read_text())["trained_at"] is None


def test_raises_when_there_is_nothing_to_export(mlflow_tmp):
    with pytest.raises(RuntimeError):
        export_model(export_dir=mlflow_tmp / "models" / "model")
