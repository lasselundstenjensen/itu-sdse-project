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


def select_run_to_promote(best_run, prod_run_id):
    """Return the run id to promote, or None if production should stay."""
    train_model_score = best_run.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = best_run.get("metrics.f1_score")

    if prod_run_id is None:
        logger.info("No model in production. The trained model will be promoted.")
        return best_run["run_id"]

    prod_metrics = mlflow.get_run(prod_run_id).data.metrics
    prod_model_score = prod_metrics.get("test_f1_score") or prod_metrics.get("f1_score")

    logger.info("Comparing models:")
    logger.info("  Current best (training): {}", train_model_score)
    logger.info("  Production: {}", prod_model_score)

    if train_model_score > prod_model_score:
        logger.info("New model is better. Promoting the new model.")
        return best_run["run_id"]

    logger.info("Current production model is better. Keeping it.")
    return None


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()
