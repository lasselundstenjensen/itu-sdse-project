"""MLflow tracking, model registry, promotion and export of the servable model.

CLI:
    python -m vial_classifier.registry promote   # compare with production, promote if better
    python -m vial_classifier.registry export    # export the production model to models/model
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient

from vial_classifier import config


def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(config.MLFLOW_STATE_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=config.MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment_name)


# --- Registering a freshly trained model (end of training) --------------------

def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            details = client.get_model_version(name=model_name, version=model_version)
            status = ModelVersionStatus.from_string(details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                print(f"Model {model_name} version {model_version} is READY")
                return True
        except Exception as e:
            print(f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}")
        time.sleep(1)
    print(f"Model {model_name} version {model_version} did not become READY after {max_attempts} attempts")
    return False


def register_and_stage_model(run_id):
    """Register the run's pyfunc model and put it in Staging with the `best` alias."""
    model_uri = f"runs:/{run_id}/{config.ARTIFACT_PATH}"
    model_details = mlflow.register_model(model_uri=model_uri, name=config.MODEL_NAME)
    wait_until_ready(model_details.name, model_details.version)

    print("Model registered successfully:")
    print(f"  Name: {model_details.name}")
    print(f"  Version: {model_details.version}")
    print(f"  Stage: {model_details.current_stage}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=config.MODEL_NAME,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(config.MODEL_NAME, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(config.MODEL_NAME, model_details.version, "task", "classification")
    client.set_registered_model_alias(name=config.MODEL_NAME, alias="best", version=str(model_details.version))

    print(f"Successfully transitioned {config.MODEL_NAME} version {model_details.version} to Staging")
    pprint(dict(model_details))
    return model_details


# --- Model selection and promotion (deployment) -------------------------------

def get_production_model(client):
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    name = config.MODEL_NAME

    try:
        model = client.get_model_version_by_alias(name=name, alias="Production")
        if model:
            print("Production model name: ", name)
            print("Production model version:", model.version)
            print("Production model run id:", model.run_id)
            return f"models:/{name}@Production", model.version, model.run_id
    except Exception:
        pass

    for model in client.search_model_versions(f"name='{name}'"):
        if dict(model).get("current_stage") == "Production":
            print("Production model name: ", name)
            print("Production model version:", model.version)
            print("Production model run id:", model.run_id)
            return f"models:/{name}/{model.version}", model.version, model.run_id

    print("No model in production")
    return None, None, None


def select_run_to_promote(client):
    """Compare the best trained run with production. Return the run id to
    promote, or None if the current production model should be kept."""
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    best = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    print("Best run:", best["run_id"])
    print("Test F1-score:", best.get("metrics.test_f1_score"))

    train_score = best.get("metrics.test_f1_score")
    if train_score is None:
        train_score = best.get("metrics.f1_score")

    production_uri, _, prod_run_id = get_production_model(client)

    if production_uri is None:
        print("No model in production. The trained model will be promoted.")
        return best["run_id"]

    prod_run = mlflow.get_run(prod_run_id)
    prod_score = prod_run.data.metrics.get("test_f1_score") or prod_run.data.metrics.get("f1_score")

    print("Comparing models:")
    print(f"  Current best (training): {train_score}")
    print(f"  Production: {prod_score}")

    if train_score > prod_score:
        print("New model is better. Promoting the new model.")
        return best["run_id"]

    print("Current production model is better. Keeping it.")
    return None


def wait_for_stage(client, model_name, version, stage="Staging", max_attempts=20):
    for _ in range(max_attempts):
        details = dict(client.get_model_version(name=model_name, version=version))
        if details["current_stage"] == stage:
            print(f"Transition completed to {stage}")
            return True
        print(f"Waiting for transition to {stage} (current: {details['current_stage']})")
        time.sleep(2)
    return False


def promote_model(client, run_id):
    """Promote the registered version of `run_id` to Production (via Staging)."""
    name = config.MODEL_NAME

    versions = [mv for mv in client.search_model_versions(f"name='{name}'") if mv.run_id == run_id]
    if not versions:
        print("Run was not registered yet, registering it now")
        version = mlflow.register_model(f"runs:/{run_id}/{config.ARTIFACT_PATH}", name=name).version
    else:
        version = versions[0].version

    print(f"Promoting {name} version {version}")

    details = dict(client.get_model_version(name=name, version=version))
    if details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=name, version=version, stage="Staging", archive_existing_versions=True,
        )
        wait_for_stage(client, name, version, "Staging")

    client.transition_model_version_stage(
        name=name, version=version, stage="Production", archive_existing_versions=True,
    )
    wait_for_stage(client, name, version, "Production")
    client.set_registered_model_alias(name=name, alias="Production", version=str(version))
    client.set_model_version_tag(name, version, "deployed_to", "production")
    print(f"{name} version {version} is now in Production")
    return version


def promote_best_model():
    """Select the best run and promote it if it beats production.

    Returns the promoted version, or None if production was kept.
    """
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    client = MlflowClient()

    run_id = select_run_to_promote(client)
    print(f"Selected run: {run_id}")
    if run_id is None:
        return None
    return promote_model(client, run_id)


# --- Export (what gets baked into the Docker image) ---------------------------

def resolve_model(client):
    """Resolve which model to serve: Production alias, else the newest
    registered version, else the latest run's artifact.

    Returns (model_uri, description, version_or_None).
    """
    name = config.MODEL_NAME

    try:
        mv = client.get_model_version_by_alias(name=name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{name} version {mv.version} (Production alias)", mv.version
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return source, f"{name} version {mv.version} (newest registered; no Production alias)", mv.version
    except Exception:
        pass

    runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME], max_results=1,
                              order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError(
            f"No registered {name} model and no runs in experiment {config.EXPERIMENT_NAME} - nothing to export"
        )
    return f"runs:/{runs.iloc[0].run_id}/{config.ARTIFACT_PATH}", "latest run artifact (nothing registered)", None


def export_model(export_dir=config.EXPORT_DIR):
    """Download the model to a plain directory and write its version info."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    client = MlflowClient()

    model_uri, description, resolved_version = resolve_model(client)
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    Path(export_dir).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if Path(export_dir).exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)
    print(f"  Exported to: {Path(export_dir).resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=config.MODEL_NAME, alias="Production")
        start_time = client.get_run(prod_mv.run_id).info.start_time / 1000
        trained_at = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": str(resolved_version or "unknown"), "trained_at": None}

    with open(Path(export_dir) / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    print("Version info:", version_info)
    return version_info


def main():
    parser = argparse.ArgumentParser(description="Model registry operations.")
    parser.add_argument("command", choices=["promote", "export"])
    args = parser.parse_args()

    if args.command == "promote":
        promote_best_model()
    else:
        export_model()


if __name__ == "__main__":
    main()