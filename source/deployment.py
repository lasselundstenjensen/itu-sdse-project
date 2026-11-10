import os
import json
import time
import pandas as pd
import numpy as np
import mlflow
import shutil
import tempfile
import atexit
import subprocess
import sys
import urllib.error
import urllib.request
import warnings
import threading
import requests
import base64
import uvicorn


from datetime import datetime, timedelta, timezone
from scipy.stats import ks_2samp
from pathlib import Path
from mlflow.tracking import MlflowClient
from pprint import pprint
from fastapi import FastAPI, Query, Request, Response



if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlflow-state", exist_ok=True)


warnings.filterwarnings('ignore')

"""
MODEL CONFIGURATION
(module level, because function defaults below - e.g. drift_default_days -
are evaluated when the functions are defined)
"""

model_name = "cnn_model"
experiment_name = "vial_image_classification"
artifact_path = "model_pyfunc"

model_version = 1  # fallback, the real version is looked up from the registry below

artifact_dir = "artifacts"
mlflow_state_dir = "mlflow-state"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
inference_log_path = artifact_dir + "/inference_log.json"
model_results_path = artifact_dir + "/model_results.json"
drift_default_days = 7
drift_ks_pvalue_threshold = 0.05

# Where the exported servable model goes (this is what gets baked into the Docker image)
export_dir = "models/model"
model_path = export_dir

# Serving ports
port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"

# Where the endpoint lives in production
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"

mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"

scoring_proc = None  # set in __main__ once the scoring server is started

""" 
This function will start the tracking of the expirement
"""
def setup_mlflow_tracking(experiment_name=None):
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if experiment_name is None:
        return
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)

"""
This function will return general information on the current production model

fixes: It has 3x None in the return statement
"""
def get_production_model():
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            print('Production model name: ', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get('current_stage') == 'Production':
            print('Production model name: ', model_name)
            print('Production model version:', model.version)
            print('Production model run id:', model.run_id)
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    print('No model in production')
    return None, None, None

"""
This is looking at the status of deployment for the model?
"""
def wait_for_deployment(model_name, version, stage='Staging', max_attempts=20):
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

"""
I do not know what it does
"""
def resolve_model_uri():
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
            return source, f"{model_name} version {mv.version} (newest registered; no Production alias)"
    except Exception:
        pass

    runs = mlflow.search_runs(experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError(f"No registered {model_name} model and no runs in experiment {experiment_name} - nothing to export")
    return f"runs:/{runs.iloc[0].run_id}/{artifact_path}", "latest run artifact (nothing registered)"

"""

"""
def _start_scoring_server():
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_path,
        "-p", str(upstream_port),
        "--workers", "1",
        "--host", upstream_host,
        "--env-manager", "local",
    ], env=env)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc

"""

"""
def load_inference_logs(log_path=inference_log_path):
    """Load the inference log JSON file (may not exist yet)."""
    if not Path(log_path).exists():
        print(f"Warning: Inference log not found at {log_path}")
        return []
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                return []
            return logs
    except Exception as e:
        print(f"Warning: Failed to load inference log: {e}")
        return []

"""

"""
def filter_logs_by_time(logs, days=drift_default_days):
    """Only keep log entries from the last N days."""
    if not logs:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for log in logs:
        try:
            timestamp_str = log.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff:
                    filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered

"""

"""
def load_baseline_stats():
    """Load the baseline statistics written during training."""
    if not Path(drift_baseline_stats_path).exists():
        print(f"Warning: Baseline stats not found at {drift_baseline_stats_path}")
        return None
    try:
        with open(drift_baseline_stats_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load baseline stats: {e}")
        return None

"""

"""
def calculate_confidence_metrics(confidences):
    confidences_array = np.array(confidences)
    return {
        'mean': float(np.mean(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'std': float(np.std(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'min': float(np.min(confidences_array)) if len(confidences_array) > 0 else 0.0,
        'max': float(np.max(confidences_array)) if len(confidences_array) > 0 else 0.0,
    }

"""

"""
def check_confidence_drift(current_confidences, baseline_stats):
    """KS test of the current confidence scores against the baseline."""
    if not baseline_stats or 'confidence_statistics' not in baseline_stats:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline = baseline_stats['confidence_statistics']
    current_confidences_array = np.array(current_confidences)

    if len(current_confidences_array) < 2:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline_mean = baseline['mean']
    baseline_std = baseline['std']
    current_mean = float(np.mean(current_confidences_array))

    baseline_samples = np.random.normal(baseline_mean, baseline_std, size=len(current_confidences_array))
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_confidences_array)

    mean_diff = abs(current_mean - baseline_mean)
    ks_drift = ks_pvalue < drift_ks_pvalue_threshold
    mean_drift = mean_diff > 0.1
    drift_detected = bool(ks_drift or mean_drift)

    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'mean_diff': mean_diff,
        'current_mean': current_mean,
        'baseline_mean': baseline_mean,
        'drift_detected': drift_detected,
        'threshold_ks': drift_ks_pvalue_threshold,
        'threshold_mean': 0.1,
    }

"""

"""
def check_class_distribution_drift(predictions, baseline_stats):
    """KS test of the predicted class distribution against the baseline."""
    if not baseline_stats or 'class_distribution' not in baseline_stats:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline = baseline_stats['class_distribution']
    baseline_total = baseline['total']
    current_total = len(predictions)

    if current_total == 0:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline_prop_0 = baseline['class_0_count'] / baseline_total
    baseline_prop_1 = baseline['class_1_count'] / baseline_total
    baseline_dist = np.array([baseline_prop_0, baseline_prop_1])

    baseline_samples = np.random.choice([0, 1], size=baseline_total, p=baseline_dist)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    ks_drift = ks_pvalue < drift_ks_pvalue_threshold
    drift_detected = bool(ks_drift)

    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'drift_detected': drift_detected,
        'threshold_ks': drift_ks_pvalue_threshold,
    }

"""

"""
def detect_drift(logs, days=drift_default_days, baseline_stats=None):
    """Main drift detection: confidences + class distribution."""
    filtered_logs = filter_logs_by_time(logs, days=days)

    if not filtered_logs:
        print(f"Warning: No inference logs found in the last {days} days")
        return {
            'overall_drift_detected': False,
            'num_samples': 0,
            'confidence_drift': None,
            'class_distribution_drift': None,
            'error': 'No logs found',
        }

    confidences = []
    predictions = []
    for log in filtered_logs:
        if 'confidence' in log and log['confidence'] is not None:
            confidences.append(float(log['confidence']))
        if 'prediction' in log and log['prediction'] is not None:
            predictions.append(int(log['prediction']))

    num_samples = len(filtered_logs)

    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})

    overall_drift = (
        confidence_results.get('drift_detected', False) or
        class_dist_results.get('drift_detected', False)
    )

    return {
        'overall_drift_detected': overall_drift,
        'num_samples': num_samples,
        'confidence_drift': confidence_results,
        'class_distribution_drift': class_dist_results,
        'confidence_stats': calculate_confidence_metrics(confidences),
        'class_distribution': {
            'class_0_count': predictions.count(0),
            'class_1_count': predictions.count(1),
            'total': len(predictions),
        },
    }

"""

"""
def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{upstream_host}:{upstream_port}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

"""

"""
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(content='{"status": "error"}', status_code=503, media_type="application/json")
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")

"""

"""
def version_endpoint():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = Path(model_path) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}

"""

"""
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(
                content=resp.read(),
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(
            content=e.read(),
            status_code=e.code,
            media_type=e.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(
            content=f'{{"status": "error", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )

"""

"""
def drift(days: int = Query(default=drift_default_days, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(inference_log_path)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)

if __name__ == "__main__":

    print("Configuration loaded")

    setup_mlflow_tracking(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    """
    MODEL SELECTION
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_ids = [experiment.experiment_id]

    # The best run of the experiment is the one with the highest test F1-score
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    print("Best run:", experiment_best["run_id"])
    print("Test F1-score:", experiment_best.get("metrics.test_f1_score"))

    with open(model_results_path, "r") as f:
        model_results = json.load(f)
    pprint(model_results["classification_report"])

    client = MlflowClient()

    production_model_uri, prod_model_version, prod_model_run_id = get_production_model()

    """
    COMPARING THE PRODUCTION MODEL TO THE NEW BEST TRAINED MODEL
    """

    train_model_score = experiment_best.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.f1_score")

    model_status = {}
    run_id = None

    if production_model_uri is not None:
        prod_run = mlflow.get_run(prod_model_run_id)
        prod_model_score = prod_run.data.metrics.get("test_f1_score") or prod_run.data.metrics.get("f1_score")

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

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

    """
    Promote best model
    """

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
        wait_for_deployment(model_name, version, 'Staging')

    # And to Production (we only get here if the comparison above decided we should)
    if run_id is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        wait_for_deployment(model_name, version, 'Production')
        client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
        client.set_model_version_tag(model_name, version, "deployed_to", "production")
        print(f"{model_name} version {version} is now in Production")

    """
    Export the serving model
    """

    model_uri, description = resolve_model_uri()
    print(f"Exporting: {description}")
    print(f"  Source: {model_uri}")

    Path(export_dir).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if Path(export_dir).exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)
    print(f"  Exported to: {Path(export_dir).resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        trained_at = datetime.fromtimestamp(client.get_run(prod_mv.run_id).info.start_time / 1000, tz=timezone.utc).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": str(version), "trained_at": None}

    with open(Path(export_dir) / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    print("Version info:", version_info)

    """
    SERVE THE MODEL LOCALLY
    """
    
    scoring_proc = _start_scoring_server()
    print("Scoring server starting on port", upstream_port)


    """
    Drift detection functions 
    """

    print("Drift detection functions defined")

    app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

    # Handlers are defined at module level above; register them on the app here
    app.get("/health")(health)
    app.get("/version")(version_endpoint)
    app.post("/invocations")(invocations)
    app.post("/infer")(invocations)
    app.get("/drift")(drift)
    print("API defined")

    # uvicorn.run would block the notebook, so the API runs in a background thread
    api_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    api_server = uvicorn.Server(api_config)

    api_thread = threading.Thread(target=api_server.run, daemon=True)
    api_thread.start()
    print(f"API server starting on http://localhost:{port} (background thread)")

    for attempt in range(120):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            print(f"[{attempt}] /health -> {r.status_code} {r.text.strip()}")
            if r.status_code == 200:
                print("Model server is ready!")
                break
        except requests.RequestException:
            print(f"[{attempt}] waiting for the API server...")
        time.sleep(2)

    """
    TEST THE ENDPOINT
    """

    test_metadata = pd.read_csv("data/test/metadata.csv")
    sample_file = test_metadata.iloc[0]["filename"]
    sample_label = int(test_metadata.iloc[0]["class"])
    print(f"Sample image: {sample_file} (label={sample_label})")

    with open(Path("data/test") / sample_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {"instances": [img_b64]}

    r = requests.post(f"http://localhost:{port}/infer", json=payload, timeout=60)
    print("POST /infer ->", r.status_code)
    pprint(r.json())

    print("GET /version ->", requests.get(f"http://localhost:{port}/version").json())
    print("GET /health  ->", requests.get(f"http://localhost:{port}/health").json())

    """
    DRIFT
    """

    logs = load_inference_logs(inference_log_path)
    print(f"Logged requests: {len(logs)}")
    pprint(logs[-3:])

    baseline = load_baseline_stats()
    drift_results = detect_drift(logs, days=drift_default_days, baseline_stats=baseline)
    print("\nDrift check results:")
    pprint(drift_results)

    if drift_results["overall_drift_detected"]:
        print("\n!! Drift detected - time to retrain and redeploy")
    else:
        print("\nNo drift detected")

    # The endpoint does the same check over HTTP
    print("\nGET /drift ->")
    pprint(requests.get(f"http://localhost:{port}/drift").json())

    """
    DEPLOY
    """
    # This is what the platform team used to run by hand - now it belongs in CI/CD.
    # Nothing here is executed by the notebook.

    # !docker build -t ghcr.io/OWNER/model-serve:TAG -f docker/Dockerfile.model-serve .
    # !docker push ghcr.io/OWNER/model-serve:TAG
    #
    # # render the placeholders in the app spec and hand it to DigitalOcean
    # !doctl apps spec validate infra/digitalocean/app.yaml
    # !doctl apps create --spec infra/digitalocean/app.yaml   # first time
    # !doctl apps update APP_ID --spec infra/digitalocean/app.yaml  # after that

    """
    CLEANUP
    """
    api_server.should_exit = True
    if scoring_proc is not None and scoring_proc.poll() is None:
        scoring_proc.terminate()
        scoring_proc.wait(timeout=10)
        print("Scoring server stopped")
    print("API server stopped")










