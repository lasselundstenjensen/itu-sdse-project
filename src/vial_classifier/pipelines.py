"""End-to-end pipelines, mirroring the two original notebooks.

* ``run_training``  - ``01_training.ipynb``: data -> model -> artifacts -> MLflow
  registry (Staging).
* ``run_promotion`` - ``02_deployment.ipynb``: pick the best run, compare it
  with Production, promote if it wins, and export the servable model directory.

Both are plain functions so that CI (through Dagger) can call them via the
CLI in ``__main__``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient

from . import config
from .data import (
    calculate_image_statistics,
    create_dataloaders,
    load_image_dataset,
    pull_data,
    split_dataset,
)
from .inference import SimpleLoggingWrapper
from .model import ARCHITECTURE_NAME, SimpleCNN
from .train import (
    compute_baseline_statistics,
    evaluate_cnn_model,
    get_device,
    train_cnn_model,
    tune_threshold,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def setup_mlflow_tracking(experiment_name: str | None = config.EXPERIMENT_NAME) -> None:
    """Point MLflow at the local SQLite store and create the experiment if needed."""
    config.MLFLOW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if experiment_name is None:
        return
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=config.MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment_name)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def dvc_pull() -> None:
    """Fetch DVC-tracked metadata. Failures are logged, not raised, as in the notebook."""
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning("dvc pull failed: %s", result.stderr.strip())


# ---------------------------------------------------------------------------
# Training (01_training)
# ---------------------------------------------------------------------------
def run_training(pull: bool = True) -> dict:
    """Train the CNN, log everything to MLflow and register it in Staging.

    Returns a summary dict with the registered model name and version.
    """
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    for directory in (config.ARTIFACT_DIR, config.MLFLOW_STATE_DIR, config.RAW_IMAGE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if pull:
        dvc_pull()
        pull_data()

    samples = load_image_dataset()
    _write_json(config.CLASS_NAMES_PATH, {str(k): v for k, v in config.CLASS_NAMES.items()})

    train_samples, val_samples, test_samples = split_dataset(samples)
    train_loader, val_loader, test_loader = create_dataloaders(train_samples, val_samples, test_samples)

    image_stats = calculate_image_statistics(train_loader)
    _write_json(config.IMAGE_STATISTICS_PATH, image_stats)

    device = get_device()
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=config.TRAINING_RUN_NAME) as run:
        _log_training_params(device)

        model, _history = train_cnn_model(train_loader, val_loader, device=device)
        threshold = tune_threshold(model, val_loader, device=device)
        test_results = evaluate_cnn_model(model, test_loader, device=device, threshold=threshold)

        baseline = compute_baseline_statistics(model, val_loader, device=device, threshold=threshold)
        _write_json(config.DRIFT_BASELINE_STATS_PATH, baseline)
        _log_baseline_metrics(baseline)
        mlflow.log_artifact(str(config.DRIFT_BASELINE_STATS_PATH), "drift_detection")

        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": ARCHITECTURE_NAME,
            "input_shape": (3, *config.IMAGE_SIZE),
            "output_shape": (1,),
            "threshold": threshold,
            "num_classes": config.NUM_CLASSES,
        }, config.MODEL_CHECKPOINT_PATH)
        _write_json(config.MODEL_RESULTS_PATH, test_results)
        mlflow.log_param("classification_threshold", threshold)

        wrapper = SimpleLoggingWrapper(model=model, threshold=threshold)
        model_info = mlflow.pyfunc.log_model(artifact_path=config.ARTIFACT_PATH, python_model=wrapper)

        run_id = run.info.run_id

    version = _register_in_staging(model_info.model_uri)
    logger.info("Registered %s version %s in Staging", config.MODEL_NAME, version)
    return {"model_name": config.MODEL_NAME, "version": version, "run_id": run_id,
            "threshold": threshold, "test_f1": test_results["f1_score"]}


def _log_training_params(device: torch.device) -> None:
    mlflow.log_param("model_architecture", ARCHITECTURE_NAME)
    mlflow.log_param("batch_size", config.BATCH_SIZE)
    mlflow.log_param("num_epochs", config.NUM_EPOCHS)
    mlflow.log_param("learning_rate", config.LEARNING_RATE)
    mlflow.log_param("initial_threshold", config.THRESHOLD)
    mlflow.log_param("num_classes", config.NUM_CLASSES)
    mlflow.log_param("image_size", "x".join(map(str, config.IMAGE_SIZE)))
    mlflow.log_param("random_seed", config.RANDOM_STATE)
    mlflow.log_param("device", str(device))
    total = sum(p.numel() for p in SimpleCNN().parameters())
    mlflow.log_param("total_parameters", total)
    mlflow.log_param("trainable_parameters", sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad))


def _log_baseline_metrics(baseline: dict) -> None:
    img = baseline["image_statistics"]
    for channel in ("r", "g", "b"):
        mlflow.log_metric(f"baseline_image_mean_{channel}", img[f"mean_{channel}"])
        mlflow.log_metric(f"baseline_image_std_{channel}", img[f"std_{channel}"])
    mlflow.log_metric("baseline_class_0_count", baseline["class_distribution"]["class_0_count"])
    mlflow.log_metric("baseline_class_1_count", baseline["class_distribution"]["class_1_count"])
    mlflow.log_metric("baseline_confidence_mean", baseline["confidence_statistics"]["mean"])
    mlflow.log_metric("baseline_confidence_std", baseline["confidence_statistics"]["std"])


def _wait_until_ready(client: MlflowClient, name: str, version: str, max_attempts: int = 10) -> bool:
    for _ in range(max_attempts):
        try:
            details = client.get_model_version(name=name, version=version)
            if ModelVersionStatus.from_string(details.status) == ModelVersionStatus.READY:
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Waiting for model version: %s", exc)
        time.sleep(1)
    logger.warning("Model %s v%s did not become READY", name, version)
    return False


def _register_in_staging(model_uri: str) -> str:
    setup_mlflow_tracking(None)
    client = MlflowClient()
    details = mlflow.register_model(model_uri, name=config.MODEL_NAME)
    _wait_until_ready(client, details.name, details.version)

    client.transition_model_version_stage(
        name=config.MODEL_NAME,
        version=details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(config.MODEL_NAME, details.version, "deployed_to", "staging")
    client.set_model_version_tag(config.MODEL_NAME, details.version, "task", "classification")
    client.set_registered_model_alias(name=config.MODEL_NAME, alias="best", version=str(details.version))
    return str(details.version)


# ---------------------------------------------------------------------------
# Selection, promotion and export (02_deployment)
# ---------------------------------------------------------------------------
def select_best_run() -> pd.Series:
    """The run in the experiment with the highest test F1-score."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment {config.EXPERIMENT_NAME}")
    return runs.iloc[0]


def get_production_model(client: MlflowClient):
    """Return ``(model_version, run_id)`` for Production, or ``(None, None)``.

    Looks up the ``Production`` alias first, then falls back to the stage.
    """
    try:
        mv = client.get_model_version_by_alias(name=config.MODEL_NAME, alias="Production")
        if mv:
            return mv, mv.run_id
    except Exception:  # noqa: BLE001 - alias may not exist yet
        pass

    for mv in client.search_model_versions(f"name='{config.MODEL_NAME}'"):
        if dict(mv).get("current_stage") == "Production":
            return mv, mv.run_id
    return None, None


def should_promote(candidate_f1: float | None, production_f1: float | None) -> bool:
    """Promote if there is no production model, or the candidate scores strictly higher."""
    if production_f1 is None:
        return True
    return candidate_f1 is not None and candidate_f1 > production_f1


def _wait_for_stage(client: MlflowClient, name: str, version: str, stage: str, max_attempts: int = 20) -> bool:
    for _ in range(max_attempts):
        if dict(client.get_model_version(name=name, version=version))["current_stage"] == stage:
            return True
        time.sleep(2)
    return False


def promote_version(client: MlflowClient, version: str) -> None:
    """Move a version to Staging and then to Production, and set the Production alias."""
    current = dict(client.get_model_version(name=config.MODEL_NAME, version=version))["current_stage"]
    if current != "Staging":
        client.transition_model_version_stage(
            name=config.MODEL_NAME, version=version, stage="Staging", archive_existing_versions=True,
        )
        _wait_for_stage(client, config.MODEL_NAME, version, "Staging")

    client.transition_model_version_stage(
        name=config.MODEL_NAME, version=version, stage="Production", archive_existing_versions=True,
    )
    _wait_for_stage(client, config.MODEL_NAME, version, "Production")
    client.set_registered_model_alias(name=config.MODEL_NAME, alias="Production", version=version)
    client.set_model_version_tag(config.MODEL_NAME, version, "deployed_to", "production")
    logger.info("%s version %s is now in Production", config.MODEL_NAME, version)


def run_promotion() -> dict:
    """Select the best run, promote it if it beats Production, then export the served model."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)
    client = MlflowClient()
    best = select_best_run()
    candidate_f1 = best.get("metrics.test_f1_score")

    prod_version, prod_run_id = get_production_model(client)
    production_f1 = None
    if prod_version is not None:
        production_f1 = mlflow.get_run(prod_run_id).data.metrics.get("test_f1_score")
    promoted = should_promote(candidate_f1, production_f1)

    if promoted:
        version = _ensure_registered(client, best["run_id"])
        promote_version(client, version)
    else:
        logger.info("Production model is better (%s >= %s); keeping it", production_f1, candidate_f1)

    export = export_serving_model(client)
    return {"promoted": promoted, "candidate_f1": candidate_f1, "production_f1": production_f1, **export}


def _ensure_registered(client: MlflowClient, run_id: str) -> str:
    for mv in client.search_model_versions(f"name='{config.MODEL_NAME}'"):
        if mv.run_id == run_id:
            return str(mv.version)
    details = mlflow.register_model(f"runs:/{run_id}/{config.ARTIFACT_PATH}", name=config.MODEL_NAME)
    return str(details.version)


def resolve_model_uri(client: MlflowClient) -> tuple[str, str, str | None]:
    """Which model to export: Production alias, else newest registered, else latest run.

    Returns ``(uri, description, version_or_None)``.
    """
    try:
        mv = client.get_model_version_by_alias(name=config.MODEL_NAME, alias="Production")
        return _source_of(mv), f"{config.MODEL_NAME} version {mv.version} (Production alias)", str(mv.version)
    except Exception:  # noqa: BLE001
        pass

    versions = list(client.search_model_versions(f"name='{config.MODEL_NAME}'"))
    if versions:
        mv = max(versions, key=lambda v: int(v.version))
        return _source_of(mv), f"{config.MODEL_NAME} version {mv.version} (newest registered)", str(mv.version)

    runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME], max_results=1,
                              order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError(f"No registered {config.MODEL_NAME} model and no runs - nothing to export")
    return f"runs:/{runs.iloc[0].run_id}/{config.ARTIFACT_PATH}", "latest run artifact (nothing registered)", None


def _source_of(mv) -> str:
    return getattr(mv, "source", None) or mv.storage_location


def export_serving_model(client: MlflowClient, export_dir: Path = config.EXPORT_MODEL_DIR) -> dict:
    """Download the chosen pyfunc model to ``export_dir`` and write ``version.json``."""
    uri, description, version = resolve_model_uri(client)
    logger.info("Exporting %s from %s", description, uri)

    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=export_dir.parent) as tmp:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(str(downloaded), str(export_dir))

    version_info = _version_info(client, version)
    _write_json(export_dir / "version.json", version_info)
    return {"export_dir": str(export_dir), "version_info": version_info}


def _version_info(client: MlflowClient, version: str | None) -> dict:
    """The version and training time that the ``/version`` endpoint reports."""
    if version is None:
        return {"model_version": "unknown", "trained_at": None}
    try:
        mv = client.get_model_version(name=config.MODEL_NAME, version=version)
        started = client.get_run(mv.run_id).info.start_time
        trained_at = datetime.fromtimestamp(started / 1000, tz=timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        trained_at = None
    return {"model_version": str(version), "trained_at": trained_at}
