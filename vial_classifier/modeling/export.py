"""Export the model to serve (Production alias) from MLflow to models/model, plus version.json."""
# [CHANGED] Module docstring added. Source: notebook 02, code cell 9 ("Export the serving model").

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

from vial_classifier import config


def resolve_model_uri(client):
    # [CHANGED] Takes `client` as an argument: the notebook used the global `client = MlflowClient()`
    """Resolve which model to serve: Production alias, else the newest
    registered version, else the latest run's artifact."""
    # [CHANGED] Returns a 3rd value, the model version. The notebook's version.json fallback read the
    # global `version` from the promote cell (code cell 8), which doesn't exist in this module.
    try:
        mv = client.get_model_version_by_alias(name=config.MODEL_NAME, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{config.MODEL_NAME} version {mv.version} (Production alias)", mv.version
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return (
                source,
                f"{config.MODEL_NAME} version {mv.version} (newest registered; no Production alias)",
                mv.version,
            )
    except Exception:
        pass

    runs = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME], max_results=1, order_by=["start_time DESC"]
    )
    if runs.empty:
        raise RuntimeError(
            f"No registered {config.MODEL_NAME} model and no runs in experiment {config.EXPERIMENT_NAME} - nothing to export"
        )
    return (
        f"runs:/{runs.iloc[0].run_id}/{config.MODEL_ARTIFACT_PATH}",
        "latest run artifact (nothing registered)",
        None,
    )
    # [CHANGED] config.MODEL_ARTIFACT_PATH was the notebook global artifact_path ("model_pyfunc").


def export_model(export_dir=config.EXPORT_DIR):
    # [CHANGED] New function wrapping the loose code of code cell 9, so the pipeline can call it.
    # export_dir was the notebook global export_dir ("models/model"); now config.EXPORT_DIR, overridable for tests.
    """Download the model to serve into export_dir and write version.json next to it. Returns export_dir."""
    client = (
        MlflowClient()
    )  # [CHANGED] Created here instead of using the notebook global from code cell 6.

    model_uri, description, resolved_version = resolve_model_uri(client)
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    export_dir = Path(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=export_dir.parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), str(export_dir))
    print(f"  Exported to: {export_dir.resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=config.MODEL_NAME, alias="Production")
        trained_at = datetime.fromtimestamp(
            client.get_run(prod_mv.run_id).info.start_time / 1000, tz=UTC
        ).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": str(resolved_version), "trained_at": None}
        # [CHANGED] Was str(version), the global from the promote cell; now the version resolve_model_uri found.

    with open(export_dir / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    print("Version info:", version_info)

    return export_dir  # [CHANGED] New: returned so callers and tests know where the model went.
