from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import mlflow

from vials.config import MODELS_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, MLFLOW_STATE_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MLFLOW_ARTIFACT_LOCATION


app = typer.Typer()


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


@app.command()
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow_tracking(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: {}", mlflow.get_tracking_uri())
    


if __name__ == "__main__":
    app()
