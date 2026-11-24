import os
import sys
import time
from pprint import pprint

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient

from ..config.constants import (
    experiment_name,
    mlflow_artifact_location,
    mlflow_tracking_uri,
    model_name
)
from ..config.paths import mlflow_state_dir

mlflow.set_tracking_uri(mlflow_tracking_uri)


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


def main():
    # Fetch the most recent finished run from experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No finished runs found in experiment '{experiment_name}'.")

    latest_run = runs.iloc[0]
    run_id = latest_run["run_id"]

    # Register the model from the training run
    model_uri = f"runs:/{run_id}/model_pyfunc"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    if not wait_until_ready(model_details.name, model_details.version):
        sys.exit(1)

    print("Model registered successfully:")
    print(f"  Name: {model_details.name}")
    print(f"  Version: {model_details.version}")
    print(f"  Stage: {model_details.current_stage}")

    # Transition to staging and apply tags/aliases
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

    print(f"Successfully transitioned {model_name} version {model_details.version} to Staging")
    pprint(dict(model_details))

    # Pull logged metrics/params from the finished run for the summary
    optimal_threshold = float(latest_run.get("params.optimal_threshold", 0.0))
    test_f1 = float(latest_run.get("metrics.test_f1_score", 0.0))

    print("\n" + "=" * 50)
    print("MODEL REGISTRATION COMPLETE")
    print("=" * 50)
    print("\nSummary:")
    print(f"  Run ID: {run_id}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Test F1-score: {test_f1:.4f}")
    print(f"  Registered model: {model_name} version {model_details.version} (Staging)")


if __name__ == "__main__":
    main()
