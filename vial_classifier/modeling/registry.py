import time

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from vial_classifier import config
from vial_classifier.utils import setup_mlflow_tracking

PRODUCTION_ALIAS = "Production"


def get_client():
    """Point MLflow at the local state dir and return a client for it."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    return MlflowClient()


def wait_until_ready(client, model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    for _ in range(max_attempts):
        try:
            details = client.get_model_version(name=model_name, version=model_version)
            if ModelVersionStatus.from_string(details.status) == ModelVersionStatus.READY:
                return True
        except MlflowException as e:
            print(f"Error checking model status: {e}")
        time.sleep(1)
    return False


def register_and_stage_model(client, run_id):
    """Register the model logged by a training run and move it to Staging."""
    model_uri = f"runs:/{run_id}/{config.ARTIFACT_PATH}"
    model_details = mlflow.register_model(model_uri=model_uri, name=config.MODEL_NAME)
    wait_until_ready(client, model_details.name, model_details.version)

    client.transition_model_version_stage(
        name=config.MODEL_NAME,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(
        config.MODEL_NAME, model_details.version, "deployed_to", "staging"
    )
    client.set_model_version_tag(
        config.MODEL_NAME, model_details.version, "task", "classification"
    )
    client.set_registered_model_alias(
        name=config.MODEL_NAME, alias="best", version=str(model_details.version)
    )
    return model_details


def get_production_model(client):
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=config.MODEL_NAME, alias=PRODUCTION_ALIAS)
        if model:
            return (
                f"models:/{config.MODEL_NAME}@{PRODUCTION_ALIAS}",
                model.version,
                model.run_id,
            )
    except MlflowException:
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{config.MODEL_NAME}'"):
        if dict(model).get("current_stage") == "Production":
            return f"models:/{config.MODEL_NAME}/{model.version}", model.version, model.run_id

    return None, None, None


def get_best_run():
    """The best run of the experiment is the one with the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]


def select_run_to_promote(best_run, prod_run_id=None):
    """Return the run id to promote, or None if production is at least as good."""
    train_model_score = best_run.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = best_run.get("metrics.f1_score")

    if prod_run_id is None:
        return best_run["run_id"]

    prod_metrics = mlflow.get_run(prod_run_id).data.metrics
    prod_model_score = prod_metrics.get("test_f1_score") or prod_metrics.get("f1_score")

    if train_model_score > prod_model_score:
        return best_run["run_id"]
    return None


def wait_for_deployment(client, model_name, version, stage="Staging", max_attempts=20):
    """Wait until the model version has reached the given stage."""
    for _ in range(max_attempts):
        details = dict(client.get_model_version(name=model_name, version=version))
        if details["current_stage"] == stage:
            return True
        time.sleep(2)
    return False


def promote_model(client, run_id):
    """Promote the registered version of a run to Production (via Staging)."""
    model_name = config.MODEL_NAME

    # Find the registered version that belongs to the winning run
    versions = [
        mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id
    ]
    if not versions:
        model_details = mlflow.register_model(
            f"runs:/{run_id}/{config.ARTIFACT_PATH}", name=model_name
        )
        version = model_details.version
    else:
        version = versions[0].version

    # Always via Staging
    details = dict(client.get_model_version(name=model_name, version=version))
    if details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        wait_for_deployment(client, model_name, version, "Staging")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    wait_for_deployment(client, model_name, version, "Production")
    client.set_registered_model_alias(
        name=model_name, alias=PRODUCTION_ALIAS, version=str(version)
    )
    client.set_model_version_tag(model_name, version, "deployed_to", "production")
    return version


def select_and_promote_best_model(client=None):
    """Compare the best trained model with production and promote it if it wins.

    Returns the promoted version, or None if production was kept.
    """
    client = client or get_client()
    _, _, prod_run_id = get_production_model(client)
    run_id = select_run_to_promote(get_best_run(), prod_run_id)
    if run_id is None:
        return None
    return promote_model(client, run_id)
