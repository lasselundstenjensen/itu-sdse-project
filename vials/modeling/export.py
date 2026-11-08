from loguru import logger
import mlflow
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


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()
