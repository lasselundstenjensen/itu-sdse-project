"""Select the best trained model, promote it to Production and export it for serving."""

import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import typer
from mlflow.tracking import MlflowClient

from vial_classifier.config import EXPERIMENT_NAME, EXPORT_DIR, MODEL_ARTIFACT_PATH, MODEL_NAME
from vial_classifier.modeling.train import setup_mlflow_tracking

app = typer.Typer()


def get_best_run(experiment_name=EXPERIMENT_NAME):
    """The best run of the experiment is the one with the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_ids = [experiment.experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    print("Best run:", experiment_best["run_id"])
    print("Test F1-score:", experiment_best.get("metrics.test_f1_score"))
    return experiment_best


def get_production_model(client, model_name=MODEL_NAME):
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            print("Production model name: ", model_name)
            print("Production model version:", model.version)
            print("Production model run id:", model.run_id)
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get("current_stage") == "Production":
            print("Production model name: ", model_name)
            print("Production model version:", model.version)
            print("Production model run id:", model.run_id)
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    print("No model in production")
    return None, None, None


def select_run(experiment_best, production_model_uri, prod_model_run_id):
    """Return the run id to promote, or None if the production model is better."""
    train_model_score = experiment_best.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.f1_score")

    run_id = None

    if production_model_uri is not None:
        prod_run = mlflow.get_run(prod_model_run_id)
        prod_model_score = prod_run.data.metrics.get("test_f1_score") or prod_run.data.metrics.get(
            "f1_score"
        )

        print("Comparing models:")
        print(f"  Current best (training): {train_model_score}")
        print(f"  Production: {prod_model_score}")

        if train_model_score > prod_model_score:
            print("New model is better. Promoting the new model.")
            run_id = experiment_best["run_id"]
        else:
            print("Current production model is better. Keeping it.")
    else:
        print("No model in production. The trained model will be promoted.")
        run_id = experiment_best["run_id"]

    print(f"Selected run: {run_id}")
    return run_id


def wait_for_deployment(client, model_name, version, stage="Staging", max_attempts=20):
    status = False
    for attempt in range(max_attempts):
        model_version_details = dict(client.get_model_version(name=model_name, version=version))
        if model_version_details["current_stage"] == stage:
            print(f"Transition completed to {stage}")
            status = True
            break
        else:
            print(
                f"Waiting for transition to {stage} "
                f"(current: {model_version_details['current_stage']})"
            )
            time.sleep(2)
    return status


def promote_model(client, run_id, model_name=MODEL_NAME):
    """Move the registered version of the run via Staging to Production."""
    # Find the registered version that belongs to the winning run
    versions = [
        mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id
    ]
    if not versions:
        print("Run was not registered yet, registering it now")
        model_details = mlflow.register_model(
            f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", name=model_name
        )
        version = model_details.version
    else:
        version = versions[0].version

    print(f"Promoting {model_name} version {version}")

    # Always via Staging
    model_version_details = dict(client.get_model_version(name=model_name, version=version))
    if model_version_details["current_stage"] != "Staging":
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
    client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
    client.set_model_version_tag(model_name, version, "deployed_to", "production")
    print(f"{model_name} version {version} is now in Production")
    return version


def resolve_model_uri(client, model_name=MODEL_NAME, experiment_name=EXPERIMENT_NAME):
    """Resolve which model to serve: Production alias, else the newest
    registered version, else the latest run's artifact."""
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{model_name} version {mv.version} (Production alias)"
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return (
                source,
                f"{model_name} version {mv.version} (newest registered; no Production alias)",
            )
    except Exception:
        pass

    runs = mlflow.search_runs(
        experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"]
    )
    if runs.empty:
        raise RuntimeError(
            f"No registered {model_name} model and no runs in experiment {experiment_name} "
            "- nothing to export"
        )
    return f"runs:/{runs.iloc[0].run_id}/{MODEL_ARTIFACT_PATH}", "latest run artifact"


def export_model(client, version, export_dir=EXPORT_DIR, model_name=MODEL_NAME):
    """Download the served model to export_dir and write version.json for /version."""
    model_uri, description = resolve_model_uri(client, model_name)
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    export_dir = Path(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=export_dir.parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)
    print(f"  Exported to: {export_dir.resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        trained_at = datetime.fromtimestamp(
            client.get_run(prod_mv.run_id).info.start_time / 1000, tz=timezone.utc
        ).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": str(version), "trained_at": None}

    with open(export_dir / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    print("Version info:", version_info)


# ---- CLI ----
@app.command()
def main():
    """Select the best run, promote it if it beats Production, and export the served model."""
    setup_mlflow_tracking()
    client = MlflowClient()

    experiment_best = get_best_run()
    production_model_uri, prod_model_version, prod_model_run_id = get_production_model(client)
    run_id = select_run(experiment_best, production_model_uri, prod_model_run_id)

    if run_id is not None:
        version = promote_model(client, run_id)
    else:
        version = prod_model_version

    export_model(client, version)


if __name__ == "__main__":
    app()