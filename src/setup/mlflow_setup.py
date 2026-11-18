import os
from ..config.paths import mlflow_state_dir
from ..config.constants import mlflow_tracking_uri, mlflow_artifact_location, experiment_name
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)
