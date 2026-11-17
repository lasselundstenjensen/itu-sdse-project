import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import mlflow

mlflow_state_dir = "mlflow-state"
model_name = "cnn_model"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            model_version_details = client.get_model_version(name=model_name, version=model_version)
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                print(f"Model {model_name} version {model_version} is READY")
                return True
        except Exception as e:
            print(f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}")
        time.sleep(1)
    print(f"Model {model_name} version {model_version} did not become READY after {max_attempts} attempts")
    return False

def register_model(run_id):
    """Register model and promote it to Staging"""

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"runs:/{run_id}/model_pyfunc"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    if not wait_until_ready(model_details.name, model_details.version):
        raise RuntimeError(f"Model {model_details.name} version {model_details.version} is NOT READY")
    
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(model_name, model_details.version, "task", "classification")
    client.set_registered_model_alias(name=model_name, alias="best", version=str(model_details.version))
