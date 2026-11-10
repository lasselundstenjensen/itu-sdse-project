"""Model selection, promotion and export.

- pick the best run of the experiment by test F1
- compare it with the current Production model
- promote the winner via Staging -> Production (+ alias, tags)
- export the Production pyfunc model to `config.EXPORT_DIR` + `version.json`

CLI: `python -m vial_classifier.modeling.registry` promotes and exports the model.
The exported directory is what CI uploads as the `model` artifact.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile
import time

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import typer

from vial_classifier import config
from vial_classifier.modeling.train import setup_mlflow_tracking

app = typer.Typer()


def get_best_run(experiment_name=config.EXPERIMENT_NAME):
    """The best run of the experiment is the one with the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_best = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    logger.info(
        f"Best run: {experiment_best['run_id']} "
        f"(test F1-score: {experiment_best.get('metrics.test_f1_score')})"
    )
    return experiment_best


def get_production_model(client, model_name=config.MODEL_NAME):
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            logger.info(
                f"Production model: {model_name} version {model.version} (run {model.run_id})"
            )
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:  # noqa: BLE001, S110 - kept exactly as in the notebook
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get("current_stage") == "Production":
            logger.info(
                f"Production model: {model_name} version {model.version} (run {model.run_id})"
            )
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    logger.info("No model in production")
    return None, None, None


def _f1_score_of(metrics):
    """Test F1 of a run's metrics, falling back to the legacy `f1_score` metric."""
    return metrics.get("test_f1_score") or metrics.get("f1_score")


def select_run_to_promote(experiment_best, prod_model_run_id):
    """Decide which run to promote: the best run, if it beats production.

    Returns the run id of the winner, or None if the current production model is better.
    """
    train_model_score = experiment_best.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.f1_score")

    if prod_model_run_id is None:
        logger.info("No model in production. The trained model will be promoted.")
        return experiment_best["run_id"]

    prod_run = mlflow.get_run(prod_model_run_id)
    prod_model_score = _f1_score_of(prod_run.data.metrics)
    logger.info(
        f"Comparing models: current best (training) {train_model_score}, "
        f"production {prod_model_score}"
    )

    if train_model_score > prod_model_score:
        logger.info("New model is better. Promoting the new model.")
        return experiment_best["run_id"]

    logger.info("Current production model is better. Keeping it.")
    return None


def wait_for_deployment(client, model_name, version, stage="Staging", max_attempts=20):
    """Wait until the model version has transitioned to the given stage."""
    for _ in range(max_attempts):
        model_version_details = dict(client.get_model_version(name=model_name, version=version))
        if model_version_details["current_stage"] == stage:
            logger.info(f"Transition completed to {stage}")
            return True
        logger.info(
            f"Waiting for transition to {stage} (current: {model_version_details['current_stage']})"
        )
        time.sleep(2)
    return False


def promote_model(
    client,
    run_id,
    model_name=config.MODEL_NAME,
    artifact_path=config.MODEL_ARTIFACT_PATH,
):
    """Promote the model of a run to Production (always via Staging). Returns its version.

    `run_id` is None when the comparison decided to keep the current production model;
    the notebook still looks the run up and moves it to Staging in that case, and only
    the Production transition is guarded.
    """
    # Find the registered version that belongs to the winning run
    versions = [
        mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id
    ]
    if not versions:
        logger.info("Run was not registered yet, registering it now")
        model_details = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name=model_name)
        version = model_details.version
    else:
        version = versions[0].version

    logger.info(f"Promoting {model_name} version {version}")

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

    # And to Production (we only get here if the comparison decided we should)
    if run_id is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        wait_for_deployment(client, model_name, version, "Production")
        client.set_registered_model_alias(
            name=model_name, alias="Production", version=str(version)
        )
        client.set_model_version_tag(model_name, version, "deployed_to", "production")
        logger.info(f"{model_name} version {version} is now in Production")
    return version


def resolve_model_uri(
    client,
    model_name=config.MODEL_NAME,
    experiment_name=config.EXPERIMENT_NAME,
    artifact_path=config.MODEL_ARTIFACT_PATH,
):
    """Resolve which model to serve: Production alias, else the newest registered
    version, else the latest run's artifact."""
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{model_name} version {mv.version} (Production alias)"
    except Exception:  # noqa: BLE001, S110 - kept exactly as in the notebook
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
    except Exception:  # noqa: BLE001, S110 - kept exactly as in the notebook
        pass

    runs = mlflow.search_runs(
        experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"]
    )
    if runs.empty:
        raise RuntimeError(
            f"No registered {model_name} model and no runs in experiment {experiment_name} "
            "- nothing to export"
        )
    return (
        f"runs:/{runs.iloc[0].run_id}/{artifact_path}",
        "latest run artifact (nothing registered)",
    )


def export_model(
    client,
    export_dir=config.EXPORT_DIR,
    model_name=config.MODEL_NAME,
    fallback_version=None,
):
    """Download the serving pyfunc model to `export_dir` and write its `version.json`.

    The serving image needs the model as a plain directory (no MLflow server involved
    at serving time). `version.json` feeds the `/version` endpoint of the service.
    """
    export_dir = Path(export_dir)
    model_uri, description = resolve_model_uri(client, model_name)
    logger.info(f"Exporting: {description} (source: {model_uri})")

    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=export_dir.parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)
    logger.info(f"Exported to: {export_dir.resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        start_time = client.get_run(prod_mv.run_id).info.start_time
        trained_at = datetime.fromtimestamp(start_time / 1000, tz=UTC).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:  # noqa: BLE001 - kept exactly as in the notebook
        version_info = {"model_version": str(fallback_version), "trained_at": None}

    with open(export_dir / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    logger.info(f"Version info: {version_info}")
    return version_info


@app.command()
def main(export_dir: Path = config.EXPORT_DIR):
    """Select, promote and export the best model to models/model."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    client = MlflowClient()

    experiment_best = get_best_run()
    _, _, prod_model_run_id = get_production_model(client)

    run_id = select_run_to_promote(experiment_best, prod_model_run_id)
    version = promote_model(client, run_id)
    export_model(client, export_dir=export_dir, fallback_version=version)


if __name__ == "__main__":
    app()
