import os

import mlflow
import torch

from vial_classifier import config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(config.mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=config.mlflow_artifact_location
        )
    mlflow.set_experiment(experiment_name)

    setup_mlflow_tracking(experiment_name)
