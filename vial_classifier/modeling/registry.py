import time
from pprint import pprint

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient

from vial_classifier.config import MODEL_ARTIFACT_PATH, MODEL_NAME


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            model_version_details = client.get_model_version(
                name=model_name, version=model_version
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                print(f"Model {model_name} version {model_version} is READY")
                return True
        except Exception as e:
            print(f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}")
        time.sleep(1)
    print(
        f"Model {model_name} version {model_version} did not become READY "
        f"after {max_attempts} attempts"
    )
    return False


def register_model(run_id, model_name=MODEL_NAME):
    """Register the run's pyfunc model and move it to Staging with the 'best' alias."""
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)

    print("Model registered successfully:")
    print(f"  Name: {model_details.name}")
    print(f"  Version: {model_details.version}")
    print(f"  Stage: {model_details.current_stage}")

    # Transition to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(model_name, model_details.version, "task", "classification")
    client.set_registered_model_alias(
        name=model_name, alias="best", version=str(model_details.version)
    )

    print(f"Successfully transitioned {model_name} version {model_details.version} to Staging")
    pprint(dict(model_details))
    return model_details