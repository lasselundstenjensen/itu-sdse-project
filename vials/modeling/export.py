from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
import typer

from vials.config import (
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_STATE_DIR,
    MLFLOW_TRACKING_URI,
)

app = typer.Typer()


# TODO: import from vials.modeling.train once feat/train-modules is merged
def setup_mlflow_tracking(experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    MLFLOW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment_name)


def get_best_run(experiment_name):
    """Return the run with the highest test F1-score in the experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    best_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    logger.info("Best run: {}", best_run["run_id"])
    logger.info("Test F1-score: {}", best_run.get("metrics.test_f1_score"))
    return best_run


def get_production_model(client, model_name):
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            logger.info("Production model name: {}", model_name)
            logger.info("Production model version: {}", model.version)
            logger.info("Production model run id: {}", model.run_id)
            return f"models:/{model_name}@Production", model.version, model.run_id
    except MlflowException:
        logger.info("No Production alias, checking stages instead")

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get("current_stage") == "Production":
            logger.info("Production model name: {}", model_name)
            logger.info("Production model version: {}", model.version)
            logger.info("Production model run id: {}", model.run_id)
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    logger.info("No model in production")
    return None, None, None


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()
