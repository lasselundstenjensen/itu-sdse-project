import sys
import time

import mlflow
from mlflow.tracking import MlflowClient

from ..config import MODEL_NAME
from ..utils import print_section_header


def wait_for_deployment(model_name, model_version, stage="Staging", max_attempts=20, interval=2):
    client = MlflowClient()

    for attempt in range(max_attempts):
        model_version_details = dict(
            client.get_model_version(name=model_name, version=model_version)
        )
        current_stage = model_version_details.get("current_stage", "None")

        if current_stage == stage:
            print(f"Transition completed to {stage}")
            return True
        else:
            print(f"Waiting for transition to {stage} (current: {current_stage}, attempt {attempt + 1}/{max_attempts})")

        time.sleep(interval)

    print(f"Model did not reach {stage} stage after {max_attempts} attempts")
    return False


def transition_to_staging(model_name=None, model_version=1, archive_existing=True):
    if model_name is None:
        model_name = MODEL_NAME

    print_section_header("MODEL DEPLOYMENT")
    print(f"Transitioning {model_name} version {model_version} to Staging...")

    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging",
        archive_existing_versions=archive_existing,
    )

    deployment_status = wait_for_deployment(model_name, model_version, stage="Staging")

    if deployment_status:
        print(f"Successfully transitioned {model_name} version {model_version} to Staging")
    else:
        print(f"Failed to transition {model_name} version {model_version} to Staging")

    return deployment_status


def transition_to_production(model_name=None, model_version=1, from_stage="Staging", archive_existing=True):
    if model_name is None:
        model_name = MODEL_NAME

    print_section_header("MODEL PROMOTION TO PRODUCTION")
    print(f"Transitioning {model_name} version {model_version} to Production...")

    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=archive_existing,
    )

    deployment_status = wait_for_deployment(model_name, model_version, stage="Production")

    if deployment_status:
        print(f"Successfully transitioned {model_name} version {model_version} to Production")
    else:
        print(f"Failed to transition {model_name} version {model_version} to Production")

    return deployment_status


def set_model_alias(model_name=None, model_version=1, alias="best"):
    if model_name is None:
        model_name = MODEL_NAME

    print(f"Setting alias '{alias}' for {model_name} version {model_version}...")

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(model_version),
    )
    print(f"Successfully set alias '{alias}' for {model_name} version {model_version}")
    return True


def deploy_model(model_name=None, model_version=1, transition_to_prod=False):
    if model_name is None:
        model_name = MODEL_NAME

    staging_success = transition_to_staging(model_name, model_version)

    if not staging_success:
        return False

    if transition_to_prod:
        return transition_to_production(model_name, model_version)

    return True


if __name__ == "__main__":
    print("Running deployment module...")
    success = deploy_model()
    if success:
        print("\nDeployment successful!")
    else:
        print("\nDeployment failed!")
        sys.exit(1)
