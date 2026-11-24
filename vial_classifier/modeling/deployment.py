"""Select, promote, and export the model from the MLflow experiment."""

import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import tempfile
import time
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

from vial_classifier.config import (
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_MODEL_ARTIFACT_PATH,
    MLFLOW_STATE_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_EXPORT_DIR,
    MODEL_NAME,
    MODEL_RESULTS_PATH,
)


def setup_mlflow_tracking(experiment_name=None):
    """Configure local MLflow tracking, following Notebook 2."""
    MLFLOW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=MLFLOW_ARTIFACT_LOCATION,
        )
    mlflow.set_experiment(experiment_name)


def get_best_run():
    """Return the run with the highest test F1, as in Notebook 2."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_ids = [experiment.experiment_id]

    return mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]


def display_model_results():
    """Display the saved classification report, as Notebook 2 does."""
    with open(MODEL_RESULTS_PATH, "r", encoding="utf-8") as results_file:
        model_results = json.load(results_file)
    pprint(model_results["classification_report"])


def get_production_model(client):
    """Get Production by alias, then fall back to the Production stage."""
    try:
        model = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="Production",
        )
        if model:
            print("Production model name:", MODEL_NAME)
            print("Production model version:", model.version)
            print("Production model run id:", model.run_id)
            return (
                f"models:/{MODEL_NAME}@Production",
                model.version,
                model.run_id,
            )
    except Exception:
        pass

    for model in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if dict(model).get("current_stage") == "Production":
            print("Production model name:", MODEL_NAME)
            print("Production model version:", model.version)
            print("Production model run id:", model.run_id)
            return (
                f"models:/{MODEL_NAME}/{model.version}",
                model.version,
                model.run_id,
            )

    print("No model in production")
    return None, None, None


def compare_models(experiment_best, production_model_uri, production_run_id):
    """Choose the candidate run using Notebook 2's F1 comparison."""
    train_model_score = experiment_best.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.f1_score")

    model_status = {}
    run_id = None

    if production_model_uri is not None:
        production_run = mlflow.get_run(production_run_id)
        production_model_score = (
            production_run.data.metrics.get("test_f1_score")
            or production_run.data.metrics.get("f1_score")
        )

        model_status["current"] = train_model_score
        model_status["prod"] = production_model_score

        print("Comparing models:")
        print(f"  Current best (training): {train_model_score}")
        print(f"  Production: {production_model_score}")

        if train_model_score > production_model_score:
            print("New model is better. Promoting the new model.")
            run_id = experiment_best["run_id"]
        else:
            print("Current production model is better. Keeping it.")
    else:
        print("No model in production. The trained model will be promoted.")
        run_id = experiment_best["run_id"]

    print(f"Selected run: {run_id}")
    return run_id, model_status


def wait_for_deployment(
    client,
    model_name,
    version,
    stage="Staging",
    max_attempts=20,
):
    """Poll the registry for a stage and return the notebook's status boolean."""
    status = False

    for attempt in range(max_attempts):
        model_version_details = dict(
            client.get_model_version(name=model_name, version=version)
        )
        if model_version_details["current_stage"] == stage:
            print(f"Transition completed to {stage}")
            status = True
            break

        print(
            "Waiting for transition to "
            f"{stage} (current: {model_version_details['current_stage']})"
        )
        time.sleep(2)

    return status


def promote_model(client, run_id):
    """Register and promote the selected run using Notebook 2's sequence."""
    version = None

    versions = [
        model_version
        for model_version in client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
        if model_version.run_id == run_id
    ]

    if not versions:
        print("Run was not registered yet, registering it now")
        model_details = mlflow.register_model(
            f"runs:/{run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}",
            name=MODEL_NAME,
        )
        version = model_details.version
    else:
        version = versions[0].version

    print(f"Promoting {MODEL_NAME} version {version}")

    model_version_details = dict(
        client.get_model_version(name=MODEL_NAME, version=version)
    )

    if model_version_details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        wait_for_deployment(client, MODEL_NAME, version, "Staging")

    if run_id is not None:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        wait_for_deployment(client, MODEL_NAME, version, "Production")
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=str(version),
        )
        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "deployed_to",
            "production",
        )
        print(f"{MODEL_NAME} version {version} is now in Production")

    return version


def resolve_model_uri(client):
    """Resolve the model URI in Notebook 2's fallback order."""
    try:
        model_version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="Production",
        )
        source = (
            getattr(model_version, "source", None)
            or model_version.storage_location
        )
        return (
            source,
            f"{MODEL_NAME} version {model_version.version} (Production alias)",
        )
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            model_version = max(versions, key=lambda item: int(item.version))
            source = (
                getattr(model_version, "source", None)
                or model_version.storage_location
            )
            return (
                source,
                f"{MODEL_NAME} version {model_version.version} "
                "(newest registered; no Production alias)",
            )
    except Exception:
        pass

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        max_results=1,
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError(
            f"No registered {MODEL_NAME} model and no runs "
            f"in experiment {EXPERIMENT_NAME} - nothing to export"
        )

    run_id = runs.iloc[0].run_id
    return (
        f"runs:/{run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}",
        "latest run artifact (nothing registered)",
    )


def export_serving_model(client, version):
    """Export the selected model and write version information."""
    model_uri, description = resolve_model_uri(client)
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    export_dir = Path(MODEL_EXPORT_DIR)
    export_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=export_dir.parent) as temporary_dir:
        download_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=temporary_dir,
        )
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)

    try:
        production_model = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="Production",
        )
        trained_at = datetime.fromtimestamp(
            client.get_run(production_model.run_id).info.start_time / 1000,
            tz=timezone.utc,
        ).isoformat()
        version_info = {
            "model_version": str(production_model.version),
            "trained_at": trained_at,
        }
    except Exception:
        version_info = {
            "model_version": str(version),
            "trained_at": None,
        }

    with open(export_dir / "version.json", "w", encoding="utf-8") as version_file:
        json.dump(version_info, version_file, indent=2)

    print("Version info:", version_info)
    return version_info


def prepare_serving_model():
    """Run the Notebook 2 selection, promotion, and export sequence."""
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("mlflow-state", exist_ok=True)

    setup_mlflow_tracking(EXPERIMENT_NAME)

    experiment_best = get_best_run()
    print("Best run:", experiment_best["run_id"])
    print("Test F1-score:", experiment_best.get("metrics.test_f1_score"))

    display_model_results()

    client = MlflowClient()
    production_model_uri, _, production_run_id = get_production_model(client)
    run_id, _ = compare_models(
        experiment_best,
        production_model_uri,
        production_run_id,
    )
    version = promote_model(client, run_id)
    return export_serving_model(client, version)


if __name__ == "__main__":
    prepare_serving_model()