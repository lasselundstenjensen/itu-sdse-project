import argparse
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from .config import (
    MODEL_NAME,
    ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    ARTIFACT_PATH,
    INFERENCE_LOG_PATH,
    EXPERIMENT_NAME,
    PYTORCH_MODEL_PATH,
)
from .deployment.logging_wrapper import SimpleLoggingWrapper
from .utils import print_section_header


def get_best_model_from_registry(model_name=None, stage="Staging"):
    if model_name is None:
        model_name = MODEL_NAME

    print_section_header("GETTING BEST MODEL FROM REGISTRY")
    print(f"Looking for model '{model_name}' in stage '{stage}'...")

    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")

    filtered_versions = []
    for mv in model_versions:
        mv_dict = dict(mv)
        if mv_dict.get("current_stage") == stage:
            version_num = int(mv_dict["version"])
            filtered_versions.append((version_num, mv_dict))

    if not filtered_versions:
        print(f"No model found in '{stage}' stage for '{model_name}'")
        return None, None, None

    filtered_versions.sort(reverse=True, key=lambda x: x[0])
    latest_version_num, latest_mv = filtered_versions[0]

    run_id = latest_mv.get("run_id")
    model_uri = f"models:/{model_name}/{latest_version_num}"

    print(f"Found best model in registry:")
    print(f"  Model: {model_name}")
    print(f"  Version: {latest_version_num}")
    print(f"  Stage: {stage}")
    print(f"  Run ID: {run_id}")
    print(f"  URI: {model_uri}")

    return model_uri, run_id, latest_version_num


def get_best_run_from_experiment(experiment_name=None, metric="f1_score"):
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME

    print_section_header("GETTING BEST RUN FROM EXPERIMENT")
    print(f"Looking for best run in experiment '{experiment_name}' by metric '{metric}'...")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None, None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        print(f"No runs found in experiment '{experiment_name}'")
        return None, None

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    metric_value = best_run[f"metrics.{metric}"]

    print(f"Found best run:")
    print(f"  Run ID: {run_id}")
    print(f"  {metric}: {metric_value}")

    return run_id, metric_value


def load_model_from_run(run_id):
    print_section_header("LOADING MODEL FROM RUN")
    print(f"Loading model from run: {run_id}")

    if PYTORCH_MODEL_PATH.exists():
        print(f"Loading from local artifact: {PYTORCH_MODEL_PATH}")
        checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location="cpu", weights_only=False)
        from .models.train import SimpleCNN
        model = SimpleCNN()
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Successfully loaded model from local artifacts")
        return model

    run = mlflow.get_run(run_id)
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/{ARTIFACT_PATH}",
        dst_path=str(ARTIFACT_DIR)
    )

    model_file = Path(local_path) / "vial_cnn_model.pth"
    if model_file.exists():
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        from .models.train import SimpleCNN
        model = SimpleCNN()
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Successfully loaded model from downloaded artifacts")
        return model

    print(f"Could not find model file in artifacts")
    return None


def create_logging_wrapper(model, log_path=None, threshold=0.5, mlflow_run_id=None):
    print_section_header("CREATING LOGGING WRAPPER")

    if log_path is None:
        log_path = INFERENCE_LOG_PATH

    wrapper = SimpleLoggingWrapper(
        model=model,
        log_path=log_path,
        threshold=threshold,
        mlflow_run_id=mlflow_run_id,
    )

    print(f"Created SimpleLoggingWrapper")
    return wrapper


def deploy_wrapped_model_to_mlflow(wrapper, model_name=None, artifact_path=None, run_name="serve_model"):
    if model_name is None:
        model_name = MODEL_NAME
    if artifact_path is None:
        artifact_path = ARTIFACT_PATH

    print_section_header("DEPLOYING WRAPPED MODEL TO MLFLOW")
    print(f"Deploying {model_name} to MLflow...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        print(f"Logging wrapped model to MLflow...")
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=wrapper,
            registered_model_name=model_name,
        )

        mlflow.log_param("wrapper_type", "SimpleLoggingWrapper")
        mlflow.log_param("model_type", type(wrapper.model).__name__)
        mlflow.log_param("inference_log_path", str(wrapper.log_path))
        mlflow.log_param("threshold", wrapper.threshold)

        model_uri = f"models:/{model_name}/latest"

        print(f"Successfully deployed wrapped model to MLflow")
        print(f"  Run ID: {run_id}")
        print(f"  Model URI: {model_uri}")

        return True, run_id, model_uri


def serve_deployed_model(model_uri, port=5001, timeout=30, background=False):
    print_section_header("STARTING MLFLOW MODEL SERVER")
    print(f"Starting server for model: {model_uri}")
    print(f"Port: {port}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    server_cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", str(port),
        "--no-conda",
    ]

    print(f"Command: {' '.join(server_cmd)}")

    if background:
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
    else:
        stdout_target = subprocess.PIPE
        stderr_target = subprocess.PIPE

    server_process = subprocess.Popen(
        server_cmd,
        stdout=stdout_target,
        stderr=stderr_target,
        text=True,
    )

    print(f"Server started on port {port}, PID: {server_process.pid}")
    print(f"Model is now serving at: http://localhost:{port}/invoke")
    return True, server_process


def stop_server(server_process):
    print("Stopping model server...")
    server_process.terminate()
    server_process.wait(timeout=10)
    print("Server stopped successfully")


def deploy_and_serve(
    model_name=None,
    port=5001,
    no_serve=False,
    no_register=False,
    transition_to_prod=False,
    from_registry=True,
    from_experiment=False,
    experiment_name=None,
    metric="f1_score",
    stage="Staging",
    timeout=30,
    background=False,
):
    if model_name is None:
        model_name = MODEL_NAME
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME

    print_section_header("DEPLOYMENT AND SERVING WORKFLOW")

    # Step 1: Get best model
    print("\n[Step 1] Getting best model...")
    model_uri, run_id, version = None, None, None

    if from_registry:
        model_uri, run_id, version = get_best_model_from_registry(model_name, stage)

    if model_uri is None and from_experiment:
        print(f"No model found in registry, trying experiment '{experiment_name}'...")
        run_id, _ = get_best_run_from_experiment(experiment_name, metric)

    if model_uri is None and run_id is None:
        print("ERROR: Could not find model in registry or experiment.")
        print("Please run the training pipeline first: python -m src.pipeline")
        return None

    # Step 2: Load the model
    print("\n[Step 2] Loading model...")
    model = None

    if model_uri and model_uri.startswith("models:/"):
        try:
            model = mlflow.pytorch.load_model(model_uri)
            print(f"Loaded model from registry (pytorch): {model_uri}")
        except Exception as e:
            print(f"Pytorch load failed: {e}")
            if run_id:
                model = load_model_from_run(run_id)
                print(f"Loaded model from run: {run_id}")
    elif run_id:
        model = load_model_from_run(run_id)
        print(f"Loaded model from run: {run_id}")

    if model is None:
        print("Failed to load model")
        return None

    print(f"Model loaded successfully: {type(model).__name__}")

    # Step 3: Create logging wrapper
    print("\n[Step 3] Creating logging wrapper...")
    wrapper = create_logging_wrapper(model, mlflow_run_id=run_id)

    # Step 4: Deploy to MLflow
    print("\n[Step 4] Deploying wrapped model to MLflow...")
    success, mlflow_run_id, mlflow_model_uri = deploy_wrapped_model_to_mlflow(
        wrapper,
        model_name=model_name if not no_register else None,
    )

    if not success:
        print("Failed to deploy to MLflow")
        return None

    # Step 5: Optionally transition to Production
    if transition_to_prod and not no_register:
        print("\n[Step 5] Transitioning to Production...")
        from .deployment.deploy import transition_to_production as _transition_to_production
        client = MlflowClient()
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        if latest_versions:
            latest_version = dict(latest_versions[0])["version"]
            _transition_to_production(
                model_name=model_name,
                model_version=latest_version,
                from_stage=stage,
            )

    # Step 6: Start model server
    server_process = None
    if not no_serve:
        print("\n[Step 6] Starting model server...")
        server_success, server_process = serve_deployed_model(
            model_uri=mlflow_model_uri,
            port=port,
            timeout=timeout,
            background=background,
        )

        if not server_success:
            print("Failed to start server")
            return None

    print(f"\nModel deployed and server started!")
    print(f"  Model URI: {mlflow_model_uri}")
    print(f"  Server port: {port}")
    print(f"  Inference endpoint: http://localhost:{port}/invoke")
    print(f"  Inference logs: {INFERENCE_LOG_PATH}")

    return {
        "success": True,
        "model_uri": mlflow_model_uri,
        "run_id": mlflow_run_id,
        "server_process": server_process,
        "port": port,
        "background": background,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Deploy the best model from MLflow Registry or experiment and serve it"
    )

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--metric", type=str, default="f1_score")
    parser.add_argument("--from-registry", action="store_true", default=True)
    parser.add_argument("--from-experiment", action="store_true", default=False)
    parser.add_argument("--stage", type=str, default="Staging", choices=["Staging", "Production", "Archived"])
    parser.add_argument("--no-register", action="store_true", default=False)
    parser.add_argument("--transition-to-prod", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-serve", action="store_true", default=False)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--background", "-d", action="store_true", default=False)

    args = parser.parse_args()

    print("=" * 60)
    print("DEPLOYMENT AND SERVING SCRIPT")
    print("=" * 60)
    print("\nThis is Step 2 of the two-step workflow.")
    print("Step 1 (training and registration): python -m src.pipeline")
    print("Step 2 (deployment and serving): python -m src.serve_model")
    print()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    result = deploy_and_serve(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        port=args.port,
        no_serve=args.no_serve,
        no_register=args.no_register,
        transition_to_prod=args.transition_to_prod,
        from_registry=args.from_registry,
        from_experiment=args.from_experiment,
        metric=args.metric,
        stage=args.stage,
        timeout=args.timeout,
        background=args.background,
    )

    if result and result.get("success"):
        print("\n" + "=" * 60)
        print("DEPLOYMENT SUCCESSFUL")
        print("=" * 60)

        if "server_process" in result and result["server_process"]:
            server_process = result["server_process"]
            if args.background or result.get("background"):
                print(f"Model server started in background on port {args.port}")
                print(f"  PID: {server_process.pid}")
                print(f"  To stop: kill {server_process.pid}")
                sys.exit(0)
            else:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping server...")
                    stop_server(server_process)
                    print("Server stopped")
                    sys.exit(0)

        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("DEPLOYMENT FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    import torch
    main()
