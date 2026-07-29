"""
Model Registry Module

Handles MLflow model registration, versioning, and promotion workflows.
"""

import json
import time
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from ..config import (
    ARTIFACT_PATH,
    EXPERIMENT_NAME,
    MODEL_NAME,
    MODEL_RESULTS_PATH,
)
from ..utils import print_section_header


def wait_until_ready(model_name: str, model_version: str, max_attempts: int = 10) -> bool:
    """
    Wait until model version is ready in MLflow.

    Polls the model version status until it becomes READY or max_attempts
    is reached.

    Parameters
    ----------
    model_name : str
        Name of the model in MLflow.
    model_version : str
        Version of the model to wait for.
    max_attempts : int, default=10
        Maximum number of attempts to check status.

    Returns
    -------
    bool
        True if model became READY, False if max_attempts reached.
    """
    client = MlflowClient()

    for attempt in range(max_attempts):
        try:
            model_version_details = client.get_model_version(
                name=model_name,
                version=model_version,
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")

            if status == ModelVersionStatus.READY:
                print(f"Model {model_name} version {model_version} is READY")
                return True

        except Exception as e:
            print(f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}")

        time.sleep(1)

    print(f"Model {model_name} version {model_version} did not become READY after {max_attempts} attempts")
    return False


def get_experiment_results(experiment_name: str = None) -> pd.DataFrame:
    """
    Get the best experiment run based on f1_score.

    Parameters
    ----------
    experiment_name : str, optional
        Name of the MLflow experiment. Uses EXPERIMENT_NAME from config if None.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with the best run's information.

    Raises
    ------
    ValueError
        If no runs are found for the experiment.
    """
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME

    print(f"Getting best run from experiment: {experiment_name}")

    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    experiment_ids = [experiment.experiment_id]

    # Search for best run by f1_score
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    experiment_best = runs.iloc[0]
    print(f"Best run: {experiment_best['run_id']}")
    return experiment_best


def load_model_results(path: str = None) -> dict:
    """
    Load model results from JSON file.

    Parameters
    ----------
    path : str, optional
        Path to the model results JSON file. Uses MODEL_RESULTS_PATH from config if None.

    Returns
    -------
    dict
        Dictionary with model results and metrics.
    """
    if path is None:
        path = MODEL_RESULTS_PATH

    print(f"Loading model results from {path}")

    if not Path(path).exists():
        raise FileNotFoundError(f"Model results not found at {path}")

    with open(path, "r") as f:
        model_results = json.load(f)

    print(f"Loaded results for CNN model")
    return model_results


def get_production_model(model_name: str = None) -> dict:
    """
    Get the current production model from MLflow.

    Parameters
    ----------
    model_name : str, optional
        Name of the model in MLflow. Uses MODEL_NAME from config if None.

    Returns
    -------
    dict
        Dictionary with production model information, or empty dict if none.
        Keys: version, run_id, current_stage, etc.
    """
    if model_name is None:
        model_name = MODEL_NAME

    client = MlflowClient()

    # Search for production models
    prod_models = [
        dict(model)
        for model in client.search_model_versions(f"name='{model_name}'")
        if dict(model).get("current_stage") == "Production"
    ]

    if not prod_models:
        print(f"No production model found for '{model_name}'")
        return {}

    prod_model = prod_models[0]
    prod_model_version = prod_model["version"]
    prod_model_run_id = prod_model["run_id"]

    print(f"Production model found:")
    print(f"  Name: {model_name}")
    print(f"  Version: {prod_model_version}")
    print(f"  Run ID: {prod_model_run_id}")

    return prod_model


def compare_models(
    results: dict,
    experiment_best: pd.DataFrame,
    prod_model: dict = None,
) -> tuple[str, bool]:
    """
    Compare the best trained model with production model.

    Determines whether to register the new model based on f1_score comparison.

    Parameters
    ----------
    results : dict
        Dictionary with model results from local training.
    experiment_best : pd.DataFrame
        Best run from MLflow experiment.
    prod_model : dict, optional
        Production model information from get_production_model().

    Returns
    -------
    tuple[str, bool]
        - run_id of the best model to register (or None)
        - True if there's a production model to compare against
    """
    # Get current best model score from MLflow
    train_model_score = experiment_best.get("metrics.f1_score")
    
    # Also check for test_f1_score if f1_score is not available
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.test_f1_score")

    if prod_model:
        # Get production model score
        try:
            run = mlflow.get_run(prod_model["run_id"])
            prod_model_score = run.data.metrics.get("test_f1_score") or run.data.metrics.get("f1_score")

            print(f"\nComparing models:")
            print(f"  Current best (training): {train_model_score}")
            print(f"  Production: {prod_model_score}")

            if train_model_score > prod_model_score:
                print("New model has better f1_score. Registering...")
                run_id = experiment_best["run_id"]
            else:
                print("Current model not better than production. Skipping registration.")
                run_id = None

        except Exception as e:
            print(f"Error getting production model metrics: {e}")
            print("Cannot compare. Registering new model as safe default.")
            run_id = experiment_best["run_id"]

    else:
        print("No production model found. Registering new model.")
        run_id = experiment_best["run_id"]

    return run_id, bool(prod_model)


def register_best_model(
    run_id: str,
    model_name: str = None,
    artifact_path: str = None,
) -> dict:
    """
    Register the best model in MLflow model registry.

    Parameters
    ----------
    run_id : str
        MLflow run ID to register as a model.
    model_name : str, optional
        Name to use for the model. Uses MODEL_NAME from config if None.
    artifact_path : str, optional
        Artifact path for the model. Uses ARTIFACT_PATH from config if None.

    Returns
    -------
    dict
        Dictionary with registered model details, or None if registration failed.
    """
    if run_id is None:
        print("No run_id provided. Skipping model registration.")
        return None

    if model_name is None:
        model_name = MODEL_NAME
    if artifact_path is None:
        artifact_path = ARTIFACT_PATH

    print(f"\nRegistering model from run {run_id}")
    print(f"  Model name: {model_name}")
    print(f"  Artifact path: {artifact_path}")

    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Wait for model to be ready
        wait_until_ready(model_details.name, model_details.version)

        print(f"Model registered successfully:")
        print(f"  Name: {model_details.name}")
        print(f"  Version: {model_details.version}")
        print(f"  Stage: {model_details.current_stage}")

        return dict(model_details)

    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def register_models() -> dict:
    """
    Complete model registration pipeline for PyTorch CNN model.

    This function combines all registration steps:
    1. Get experiment results
    2. Load model results
    3. Get production model (if any)
    4. Compare models
    5. Register best model if better than production

    Returns
    -------
    dict
        Dictionary with registration results.
    """
    print_section_header("MODEL REGISTRATION")

    try:
        # Get best experiment run
        experiment_best = get_experiment_results()

        # Load model results
        results = load_model_results()

        # Get production model
        prod_model = get_production_model()

        # Compare and decide
        run_id, has_prod = compare_models(results, experiment_best, prod_model)

        # Register if needed
        if run_id:
            model_details = register_best_model(run_id)
        else:
            model_details = None

        return {
            "experiment_best": experiment_best,
            "results": results,
            "production_model": prod_model,
            "has_production": has_prod,
            "run_id": run_id,
            "model_details": model_details,
        }

    except Exception as e:
        print(f"Error in model registration: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    """
    Run model registration module directly.

    Usage:
        python -m src.models.registry
    """
    print("Running model registration module...")
    try:
        result = register_models()
        print("\nModel registration complete!")
        print(f"Run ID: {result.get('run_id')}")
        print(f"Model details: {result.get('model_details')}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
