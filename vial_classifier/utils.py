import os

import mlflow
import torch

from vial_classifier import config

# check if GPU is avalible else falls back to CPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(config.MLFLOW_STATE_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(experiment_name)

