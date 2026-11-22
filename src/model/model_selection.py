import os
import json
import shutil
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import stat

warnings.filterwarnings('ignore')

mlflow_state_dir = "mlflow_state"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"

def setup_mlflow_tracking(experiment_name):
    """Set up MLflow tracking URI and experiment."""
    os.makedirs(mlflow_state_dir, exist_ok=True)
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())


def find_best_run(experiment_name):
    """Find the best run of the experiment based on the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")
    
    experiment_ids = [experiment.experiment_id]
    
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]

    #cheching
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
    
    run_id = experiment_best["run_id"]
    score = experiment_best.get("metrics.test_f1_score")
    if score is None:
        score = experiment_best.get("metrics.f1_score")
        
    print(f"Best run: {run_id}")
    print(f"Test F1-score: {score}")
    return experiment_best, run_id, score


def get_production_model(client, model_name):
    """Fetch the current production model, by alias or by stage."""
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            print('Production model name:', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get('current_stage') == 'Production':
            print('Production model name:', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    print('No model in production')
    return None, None, None


def wait_for_deployment(client, model_name, version, stage='Staging', max_attempts=20):
    """Wait for MLflow model version to transition stages."""
    status = False
    for attempt in range(max_attempts):
        model_version_details = dict(client.get_model_version(name=model_name, version=version))
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            print(f"Waiting for transition to {stage} (current: {model_version_details['current_stage']})")
            time.sleep(2)
    return status


def promote_model(client, model_name, run_id, artifact_path):
    """Compare and promote the best model through Staging to Production if it wins."""
    # Find the registered version that belongs to the winning run
    versions = [mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id]
    if not versions:
        print("Run was not registered yet, registering it now")
        model_details = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name=model_name)
        version = model_details.version
    else:
        version = versions[0].version

    print(f"Promoting {model_name} version {version}")

    # Always via Staging
    model_version_details = dict(client.get_model_version(name=model_name, version=version))
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        wait_for_deployment(client, model_name, version, 'Staging')

    # To Production if it beats the current prod score (or if no prod model exists)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    wait_for_deployment(client, model_name, version, 'Production')
    client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
    client.set_model_version_tag(model_name, version, "deployed_to", "production")
    print(f"{model_name} version {version} is now in Production")
    return version


def export_serving_model(client, model_name, experiment_name, artifact_path, export_dir="models/model"):
    """Export the winning MLflow pyfunc model to a directory and write version info."""
    def resolve_model_uri():
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
                return source, f"{model_name} version {mv.version} (newest registered)"
        except Exception:
            pass

        runs = mlflow.search_runs(experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"])
        if runs.empty:
            raise RuntimeError(f"No registered {model_name} model and no runs in experiment {experiment_name}")
        return f"runs:/{runs.iloc[0].run_id}/{artifact_path}", "latest run artifact"

    model_uri, description = resolve_model_uri()
    print(f"Exporting: {description}\n  Source: {model_uri}")

    #Path(export_dir).parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    #with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
        #download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        #if Path(export_dir).exists():
            ##shutil.rmtree(export_dir)
            #def handle_remove_readonly(func, path, exc_info):
                #os.chmod(path, stat.S_IWRITE)
                #func(path)

            #if os.path.exists(export_dir):
                #shutil.rmtree(export_dir, onerror=handle_remove_readonly)
    #shutil.move(str(download_dir), export_dir)
    #print(f"  Exported to: {Path(export_dir).resolve()}")
    Path(export_dir).parent.mkdir(parents=True, exist_ok=True)

    def handle_remove_readonly(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=tmp
        )

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir, onerror=handle_remove_readonly)

        shutil.move(str(download_dir), export_dir)

    print(f" Exported to: {Path(export_dir).resolve()}")

    try:
        prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        trained_at = datetime.fromtimestamp(client.get_run(prod_mv.run_id).info.start_time / 1000, tz=timezone.utc).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": "1", "trained_at": None}

    with open(Path(export_dir) / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    print("Version info:", version_info)