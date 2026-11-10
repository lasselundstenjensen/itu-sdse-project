import torch
import mlflow


from vial_classifier import config
from vial_classifier.utils import get_device, print_section_header, setup_mlflow_tracking


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)


def test_print_section_header_prints_title(capsys):
    print_section_header("TEST")
    assert "TEST" in capsys.readouterr().out

def test_setup_creates_experiment(tmp_path, monkeypatch):
    # Use a throwaway folder so the test leaves no files in the project
    state_dir = tmp_path / "mlflow-state"
    monkeypatch.setattr(config, "MLFLOW_STATE_DIR", str(state_dir))
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{state_dir}/mlflow.db")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_LOCATION", str(state_dir / "artifacts"))

    setup_mlflow_tracking("test_experiment")

    assert mlflow.get_experiment_by_name("test_experiment") is not None