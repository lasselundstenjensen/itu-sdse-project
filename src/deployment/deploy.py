"""
Deployment Module

Handles model deployment and stage transitions in MLflow.
"""

import sys
import time

import mlflow
from mlflow.tracking import MlflowClient

from ..config import MODEL_NAME
from ..utils import print_section_header


def wait_for_deployment(
    model_name: str,
    model_version: str,
    stage: str = "Staging",
    max_attempts: int = 20,
    interval: int = 2,
) -> bool:
    """
    Wait for model deployment to reach the specified stage.

    Polls the model version status until it reaches the target stage
    or max_attempts is reached.

    Parameters
    ----------
    model_name : str
        Name of the model in MLflow.
    model_version : str or int
        Version of the model to wait for.
    stage : str, default="Staging"
        Target stage to wait for (e.g., "Staging", "Production").
    max_attempts : int, default=20
        Maximum number of attempts to check status.
    interval : int, default=2
        Seconds between checks.

    Returns
    -------
    bool
        True if deployment reached target stage, False otherwise.
    """
    client = MlflowClient()
    status = False

    for attempt in range(max_attempts):
        try:
            model_version_details = dict(
                client.get_model_version(name=model_name, version=model_version)
            )
            current_stage = model_version_details.get("current_stage", "None")

            if current_stage == stage:
                print(f"Transition completed to {stage}")
                status = True
                break
            else:
                print(
                    f"Waiting for transition to {stage} "
                    f"(current: {current_stage}, attempt {attempt + 1}/{max_attempts})"
                )

        except Exception as e:
            print(f"Error checking deployment status: {e}")

        time.sleep(interval)

    if not status:
        print(
            f"Model did not reach {stage} stage after {max_attempts} attempts"
        )

    return status


def transition_to_staging(
    model_name: str = None,
    model_version: int = 1,
    archive_existing: bool = True,
) -> bool:
    """
    Transition model to Staging stage in MLflow.

    Parameters
    ----------
    model_name : str, optional
        Name of the model in MLflow. Uses MODEL_NAME from config if None.
    model_version : int, default=1
        Version of the model to transition.
    archive_existing : bool, default=True
        Whether to archive existing versions in Staging.

    Returns
    -------
    bool
        True if transition was successful, False otherwise.
    """
    if model_name is None:
        model_name = MODEL_NAME

    print_section_header("MODEL DEPLOYMENT")
    print(f"Transitioning {model_name} version {model_version} to Staging...")

    client = MlflowClient()

    # Get current model version details
    try:
        model_version_details = dict(
            client.get_model_version(name=model_name, version=model_version)
        )
        current_stage = model_version_details.get("current_stage", "None")

        if current_stage == "Staging":
            print("Model already in Staging. Skipping transition.")
            return True

        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=archive_existing,
        )

        # Wait for transition to complete
        deployment_status = wait_for_deployment(
            model_name, model_version, stage="Staging"
        )

        if deployment_status:
            print(f"Successfully transitioned {model_name} version {model_version} to Staging")
        else:
            print(f"Failed to transition {model_name} version {model_version} to Staging")

        return deployment_status

    except Exception as e:
        print(f"Error transitioning model to Staging: {e}")
        import traceback

        traceback.print_exc()
        return False


def deploy_model(
    model_name: str = None,
    model_version: int = 1,
) -> bool:
    """
    Complete deployment pipeline.

    Transitions model from None/Archive to Staging.

    Parameters
    ----------
    model_name : str, optional
        Name of the model in MLflow. Uses MODEL_NAME from config if None.
    model_version : int, default=1
        Version of the model to deploy.

    Returns
    -------
    bool
        True if deployment was successful, False otherwise.
    """
    if model_name is None:
        model_name = MODEL_NAME

    return transition_to_staging(model_name, model_version)


if __name__ == "__main__":
    """
    Run deployment module directly.

    Usage:
        python -m src.deployment.deploy
    """
    print("Running deployment module...")
    try:
        success = deploy_model()
        if success:
            print("\nDeployment successful!")
        else:
            print("\nDeployment failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
