"""
Deployment and Serving Script for Image Classification Model

This script provides a complete workflow for deploying the best model from the
MLflow Model Registry or latest experiment run, wrapping it with SimpleLoggingWrapper,
logging it to MLflow as a servable model, and starting the MLflow model server.

Workflow:
1. Get best model (from MLflow Model Registry or latest experiment run)
2. Load the PyTorch model
3. Wrap it with SimpleLoggingWrapper
4. Log the wrapped model to MLflow
5. Start MLflow model server on specified port

This is Step 2 in the two-step training/deployment workflow.
Step 1 (training and registration) is handled by: python -m src.pipeline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import mlflow
import torch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from ..config import (
    MODEL_NAME,
    ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    ARTIFACT_PATH,
    INFERENCE_LOG_PATH,
    EXPERIMENT_NAME,
    PYTORCH_MODEL_PATH,
)
from .logging_wrapper import SimpleLoggingWrapper
from ..utils import print_section_header


# =============================================================================
# MODEL RETRIEVAL FUNCTIONS
# =============================================================================

def get_best_model_from_registry(
    model_name: str = None,
    stage: str = "Staging",
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Get the best model version from MLflow Model Registry.

    Finds the latest version of the model in the specified stage.

    Parameters
    ----------
    model_name : str, optional
        Name of the model in MLflow. Uses MODEL_NAME from config if None.
    stage : str, default="Staging"
        Stage to filter by ("Staging", "Production", "Archived").

    Returns
    -------
    Tuple[Optional[str], Optional[str], Optional[int]]
        - model_uri: MLflow model URI (e.g., "models:/cnn_model/1")
        - run_id: MLflow run ID
        - version: Model version number
        Returns (None, None, None) if no model found.
    """
    if model_name is None:
        model_name = MODEL_NAME

    print_section_header("GETTING BEST MODEL FROM REGISTRY")
    print(f"Looking for model '{model_name}' in stage '{stage}'...")

    client = MlflowClient()

    try:
        # Search for all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")

        # Filter by stage and find the latest version
        filtered_versions = []
        for mv in model_versions:
            mv_dict = dict(mv)
            if mv_dict.get("current_stage") == stage:
                version_num = int(mv_dict["version"])
                filtered_versions.append((version_num, mv_dict))

        if not filtered_versions:
            print(f"No model found in '{stage}' stage for '{model_name}'")
            return None, None, None

        # Sort by version number (descending) and get the latest
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

    except Exception as e:
        print(f"Error getting best model from registry: {e}")
        return None, None, None


def get_best_run_from_experiment(
    experiment_name: str = None,
    metric: str = "f1_score",
) -> Tuple[Optional[str], Optional[float]]:
    """
    Get the best run from MLflow experiment based on specified metric.

    This is a fallback method when no model is found in the registry.

    Parameters
    ----------
    experiment_name : str, optional
        Name of the MLflow experiment. Uses EXPERIMENT_NAME from config if None.
    metric : str, default="f1_score"
        Metric to sort by (descending).

    Returns
    -------
    Tuple[Optional[str], Optional[float]]
        - run_id: MLflow run ID of the best run
        - metric_value: Value of the metric for the best run
        Returns (None, None) if no runs found.
    """
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME

    print_section_header("GETTING BEST RUN FROM EXPERIMENT")
    print(f"Looking for best run in experiment '{experiment_name}' by metric '{metric}'...")

    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None, None

        # Search for best run
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

    except Exception as e:
        print(f"Error getting best run from experiment: {e}")
        return None, None


def load_model_from_run(run_id: str) -> Optional[torch.nn.Module]:
    """
    Load a PyTorch model from a specific MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID to load the model from.

    Returns
    -------
    Optional[torch.nn.Module]
        Loaded PyTorch model, or None if loading failed.
    """
    print_section_header("LOADING MODEL FROM RUN")
    print(f"Loading model from run: {run_id}")

    try:
        # First, try to load from local artifacts path
        # This is where the training pipeline saves the model
        if PYTORCH_MODEL_PATH.exists():
            print(f"Loading from local artifact: {PYTORCH_MODEL_PATH}")
            import torch
            checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location="cpu", weights_only=False)
            
            # Create model instance and load state dict
            from ..models.train import SimpleCNN
            model = SimpleCNN()
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"Successfully loaded model from local artifacts")
            return model
        
        # If local file doesn't exist, try to download from MLflow artifacts
        # Get the run to find artifact path
        run = mlflow.get_run(run_id)
        artifact_path = ARTIFACT_PATH
        
        # Download the artifact
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_path}",
            dst_path=str(ARTIFACT_DIR)
        )
        
        # Try to load from the downloaded artifacts
        model_file = Path(local_path) / "vial_cnn_model.pth"
        if model_file.exists():
            import torch
            checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
            from ..models.train import SimpleCNN
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

    except Exception as e:
        print(f"Error loading model from run {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# WRAPPER CREATION AND DEPLOYMENT FUNCTIONS
# =============================================================================

def create_logging_wrapper(
    model: torch.nn.Module,
    log_path: Path = None,
    threshold: float = 0.5,
    mlflow_run_id: Optional[str] = None,
) -> SimpleLoggingWrapper:
    """
    Create a SimpleLoggingWrapper for the given model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to wrap.
    log_path : Path, optional
        Path to the inference log file. Uses INFERENCE_LOG_PATH from config if None.
    threshold : float, default=0.5
        Classification threshold.
    mlflow_run_id : str, optional
        MLflow run ID to log inference metrics to.

    Returns
    -------
    SimpleLoggingWrapper
        Initialized logging wrapper instance.
    """
    print_section_header("CREATING LOGGING WRAPPER")

    if log_path is None:
        log_path = INFERENCE_LOG_PATH

    # Create the wrapper
    wrapper = SimpleLoggingWrapper(
        model=model,
        log_path=log_path,
        threshold=threshold,
        mlflow_run_id=mlflow_run_id,
    )

    print(f"Created SimpleLoggingWrapper:")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Log path: {log_path}")
    print(f"  Threshold: {threshold}")
    if mlflow_run_id:
        print(f"  MLflow run ID: {mlflow_run_id}")

    return wrapper


def deploy_wrapped_model_to_mlflow(
    wrapper: SimpleLoggingWrapper,
    model_name: str = None,
    artifact_path: str = None,
    run_name: str = "deploy_and_serve",
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Log the wrapped model to MLflow as a servable model.

    Parameters
    ----------
    wrapper : SimpleLoggingWrapper
        The logging wrapper to deploy.
    model_name : str, optional
        Name to register the model under. Uses MODEL_NAME from config if None.
    artifact_path : str, optional
        Artifact path for MLflow model logging. Uses ARTIFACT_PATH from config if None.
    run_name : str, default="deploy_and_serve"
        Name for the MLflow run.

    Returns
    -------
    Tuple[bool, Optional[str], Optional[str]]
        - success: True if deployment succeeded
        - run_id: MLflow run ID
        - model_uri: MLflow model URI
    """
    if model_name is None:
        model_name = MODEL_NAME
    if artifact_path is None:
        artifact_path = ARTIFACT_PATH

    print_section_header("DEPLOYING WRAPPED MODEL TO MLFLOW")
    print(f"Deploying {model_name} to MLflow...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log the wrapped model
            print(f"Logging wrapped model to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=wrapper,
                registered_model_name=model_name,
            )

            # Log metadata
            mlflow.log_param("wrapper_type", "SimpleLoggingWrapper")
            mlflow.log_param("model_type", type(wrapper.model).__name__)
            mlflow.log_param("inference_log_path", str(wrapper.log_path))
            mlflow.log_param("threshold", wrapper.threshold)

            model_uri = f"models:/{model_name}/latest"

            print(f"Successfully deployed wrapped model to MLflow")
            print(f"  Run ID: {run_id}")
            print(f"  Model URI: {model_uri}")

            return True, run_id, model_uri

    except Exception as e:
        print(f"Error deploying wrapped model to MLflow: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


# =============================================================================
# MODEL SERVER FUNCTIONS
# =============================================================================

def serve_deployed_model(
    model_uri: str,
    port: int = 5001,
    timeout: int = 30,
    background: bool = False,
) -> Tuple[bool, Optional[subprocess.Popen]]:
    """
    Start the MLflow model server to serve the deployed model.

    Parameters
    ----------
    model_uri : str
        MLflow model URI to serve (e.g., "models:/cnn_model/latest").
    port : int, default=5001
        Port for the model server.
    timeout : int, default=30
        Maximum time to wait for server to start (in seconds).
    background : bool, default=False
        If True, run the server in the background (detached).

    Returns
    -------
    Tuple[bool, Optional[subprocess.Popen]]
        - success: True if server started successfully
        - server_process: The server subprocess, or None if failed
    """
    print_section_header("STARTING MLFLOW MODEL SERVER")
    print(f"Starting server for model: {model_uri}")
    print(f"Port: {port}")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Start the MLflow model server as a subprocess
    server_cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", str(port),
        "--no-conda",
    ]

    print(f"Command: {' '.join(server_cmd)}")

    # Redirect output based on background mode
    if background:
        # Background: redirect to /dev/null (or null on Windows)
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
        print(f"Starting server in background mode...")
    else:
        # Foreground: capture output
        stdout_target = subprocess.PIPE
        stderr_target = subprocess.PIPE

    server_process = subprocess.Popen(
        server_cmd,
        stdout=stdout_target,
        stderr=stderr_target,
        text=True,
    )

    # Wait for server to start
    print(f"Waiting for server to start (timeout: {timeout}s)...")
    max_wait = timeout
    for i in range(max_wait):
        time.sleep(1)
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                if background:
                    print(f"✓ Model server started in background on port {port}")
                else:
                    print(f"✓ Model server started successfully on port {port}")
                print(f"  PID: {server_process.pid}")
                print(f"\nModel is now serving at: http://localhost:{port}/invoke")
                return True, server_process
        except Exception as e:
            # Continue waiting
            if i % 5 == 0:
                print(f"  Waiting... ({i+1}/{max_wait})")

    # Timeout reached
    print(f"✗ Error: Model server failed to start after {max_wait} seconds")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except Exception:
        server_process.kill()
    return False, None


def check_server_health(port: int, timeout: int = 5) -> bool:
    """
    Check if the model server is healthy.

    Parameters
    ----------
    port : int
        Port of the model server.
    timeout : int, default=5
        Timeout in seconds for the health check.

    Returns
    -------
    bool
        True if server is healthy, False otherwise.
    """
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def stop_server(server_process: subprocess.Popen) -> bool:
    """
    Stop the model server.

    Parameters
    ----------
    server_process : subprocess.Popen
        The server process to stop.

    Returns
    -------
    bool
        True if server was stopped successfully.
    """
    try:
        print("Stopping model server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        print("Server stopped successfully")
        return True
    except Exception:
        print("Force killing server...")
        server_process.kill()
        return False


# =============================================================================
# MAIN WORKFLOW FUNCTIONS
# =============================================================================

def deploy_and_serve(
    model_name: str = None,
    port: int = 5001,
    no_serve: bool = False,
    no_register: bool = False,
    transition_to_prod: bool = False,
    from_registry: bool = True,
    from_experiment: bool = False,
    experiment_name: str = None,
    metric: str = "f1_score",
    stage: str = "Staging",
    timeout: int = 30,
    background: bool = False,
) -> Dict[str, Any]:
    """
    Complete deployment and serving workflow.

    This function orchestrates the entire process:
    1. Get best model from registry or experiment
    2. Load the model
    3. Wrap with SimpleLoggingWrapper
    4. Deploy to MLflow
    5. Optionally transition to Production
    6. Optionally start model server

    Parameters
    ----------
    model_name : str, optional
        Name of the model in MLflow. Uses MODEL_NAME from config if None.
    port : int, default=5001
        Port for the model server.
    no_serve : bool, default=False
        If True, skip starting the model server.
    no_register : bool, default=False
        If True, skip registering the model to MLflow (only log to run).
    transition_to_prod : bool, default=False
        If True, transition the model to Production after deployment.
    from_registry : bool, default=True
        If True, get model from MLflow Model Registry.
    from_experiment : bool, default=False
        If True, get model from latest experiment run (fallback).
    experiment_name : str, optional
        Name of the experiment. Uses EXPERIMENT_NAME from config if None.
    metric : str, default="f1_score"
        Metric to use for selecting best run from experiment.
    stage : str, default="Staging"
        Stage to look for in the registry.
    timeout : int, default=30
        Timeout for server startup.
    background : bool, default=False
        If True, run the model server in the background (detached mode).

    Returns
    -------
    Dict[str, Any]
        Dictionary with deployment results.
    """
    if model_name is None:
        model_name = MODEL_NAME
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME

    print_section_header("DEPLOYMENT AND SERVING WORKFLOW")
    print("=" * 60)

    results = {
        "success": False,
        "model_name": model_name,
        "port": port,
        "steps": {},
    }

    # Step 1: Get best model
    print("\n[Step 1] Getting best model...")
    model_uri, run_id, version = None, None, None

    if from_registry:
        model_uri, run_id, version = get_best_model_from_registry(model_name, stage)

    if model_uri is None and from_experiment:
        print(f"No model found in registry, trying experiment '{experiment_name}'...")
        run_id, _ = get_best_run_from_experiment(experiment_name, metric)
        if run_id:
            # Load model directly from run
            model = load_model_from_run(run_id)
            if model:
                results["steps"]["get_model"] = {"source": "experiment", "run_id": run_id}

    if model_uri is None and run_id is None:
        error_msg = (
            "ERROR: Could not find model in registry or experiment.\n"
            "\n"
            "Please run the training pipeline first:\n"
            "  python -m src.pipeline\n"
            "\n"
            "This will train the model and register it to MLflow Model Registry.\n"
            "After training completes, run this deployment script again."
        )
        print(error_msg)
        results["error"] = "No model found - please run training pipeline first"
        return results

    # Step 2: Load the model
    print("\n[Step 2] Loading model...")
    model = None

    if model_uri and model_uri.startswith("models:/"):
        # Load from registry
        try:
            # Try pytorch flavor first
            try:
                model = mlflow.pytorch.load_model(model_uri)
                results["steps"]["load_model"] = {"source": "registry", "model_uri": model_uri, "flavor": "pytorch"}
                print(f"Loaded model from registry (pytorch): {model_uri}")
            except Exception as pytorch_error:
                # If pytorch fails, try loading from run artifacts
                print(f"Pytorch load failed: {pytorch_error}, trying run artifacts...")
                if run_id:
                    model = load_model_from_run(run_id)
                    if model:
                        results["steps"]["load_model"] = {"source": "run_fallback", "run_id": run_id}
                        print(f"Loaded model from run: {run_id}")
                    else:
                        raise Exception(f"Failed to load from run {run_id}")
                else:
                    raise Exception(f"No run_id available to load from artifacts")
        except Exception as e:
            print(f"Error loading from registry: {e}")
            results["error"] = f"Failed to load from registry: {e}"
            return results
    elif run_id:
        # Already loaded from experiment
        if model is None:
            model = load_model_from_run(run_id)
            if model is None:
                results["error"] = f"Failed to load model from run {run_id}"
                return results
            results["steps"]["load_model"] = {"source": "run", "run_id": run_id}

    if model is None:
        results["error"] = "Failed to load model"
        return results

    print(f"✓ Model loaded successfully: {type(model).__name__}")

    # Step 3: Create logging wrapper
    print("\n[Step 3] Creating logging wrapper...")
    wrapper = create_logging_wrapper(model, mlflow_run_id=run_id)
    results["steps"]["create_wrapper"] = {"mlflow_run_id": run_id}

    # Step 4: Deploy to MLflow
    print("\n[Step 4] Deploying wrapped model to MLflow...")
    success, mlflow_run_id, mlflow_model_uri = deploy_wrapped_model_to_mlflow(
        wrapper,
        model_name=model_name if not no_register else None,
    )

    if not success:
        results["error"] = "Failed to deploy to MLflow"
        return results

    results["steps"]["deploy_mlflow"] = {
        "run_id": mlflow_run_id,
        "model_uri": mlflow_model_uri,
    }

    # Step 5: Optionally transition to Production
    if transition_to_prod and not no_register:
        print("\n[Step 5] Transitioning to Production...")
        from .deploy import transition_to_production as _transition_to_production

        # Find the latest version
        client = MlflowClient()
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        if latest_versions:
            latest_version = dict(latest_versions[0])["version"]
            prod_success = _transition_to_production(
                model_name=model_name,
                model_version=latest_version,
                from_stage=stage,
            )
            results["steps"]["transition_prod"] = prod_success
        else:
            print("No model version found for transition")

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
            results["error"] = "Failed to start server"
            return results

        results["steps"]["serve"] = {
            "port": port,
            "process_id": server_process.pid if server_process else None,
        }
        results["server_process"] = server_process

    # Success!
    results["success"] = True
    results["model_uri"] = mlflow_model_uri
    results["run_id"] = mlflow_run_id
    results["background"] = background

    if server_process:
        results["message"] = (
            f"Model deployed and server started!\n"
            f"  Model URI: {mlflow_model_uri}\n"
            f"  Server port: {port}\n"
            f"  Inference endpoint: http://localhost:{port}/invoke\n"
            f"  Inference logs: {INFERENCE_LOG_PATH}"
        )
    else:
        results["message"] = (
            f"Model deployed to MLflow!\n"
            f"  Model URI: {mlflow_model_uri}\n"
            f"  To start server: mlflow models serve -m {mlflow_model_uri} -p {port}"
        )

    return results


def main():
    """Main entry point for the deployment and serving script."""
    parser = argparse.ArgumentParser(
        description="Deploy the best model from MLflow Registry or experiment and serve it"
    )

    # Model selection arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=f"Name of the model in MLflow (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=f"Name of the experiment (default: {EXPERIMENT_NAME})",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_score",
        help="Metric to use for selecting best run (default: f1_score)",
    )

    # Source selection
    parser.add_argument(
        "--from-registry",
        action="store_true",
        default=True,
        help="Get model from MLflow Model Registry (default)",
    )
    parser.add_argument(
        "--from-experiment",
        action="store_true",
        default=False,
        help="Get model from latest experiment run",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        choices=["Staging", "Production", "Archived"],
        help="Stage to look for in registry (default: Staging)",
    )

    # Deployment options
    parser.add_argument(
        "--no-register",
        action="store_true",
        default=False,
        help="Skip registering the model to MLflow Model Registry",
    )
    parser.add_argument(
        "--transition-to-prod",
        action="store_true",
        default=False,
        help="Transition the model to Production after Staging",
    )

    # Server options
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port for the model server (default: 5001)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        default=False,
        help="Deploy to MLflow only, skip starting the model server",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for server startup in seconds (default: 30)",
    )
    parser.add_argument(
        "--background",
        "-d",
        action="store_true",
        default=False,
        help="Run the model server in the background (detached mode)",
    )

    args = parser.parse_args()

    # Check if MLflow server is accessible
    print("=" * 60)
    print("DEPLOYMENT AND SERVING SCRIPT")
    print("=" * 60)
    print("\nThis is Step 2 of the two-step workflow.")
    print("Step 1 (training and registration): python -m src.pipeline")
    print("Step 2 (deployment and serving): python -m src.deployment.deploy_and_serve")
    print()
    
    # Verify MLflow tracking URI is accessible
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        # Try a simple ping by getting the current tracking URI
        current_uri = mlflow.get_tracking_uri()
        # Try to search for registered models as a simple connectivity check
        try:
            client.search_registered_models()
        except Exception:
            # If that fails, try search_model_versions
            client.search_model_versions("name='test'")
        print(f"✓ MLflow server is running at: {current_uri}")
    except Exception as e:
        print(f"\n✗ ERROR: Cannot connect to MLflow server at {MLFLOW_TRACKING_URI}")
        print(f"  Error: {e}")
        print("\nPlease start the MLflow server first:")
        print(f"  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000")
        print("\nThen run this script again.")
        sys.exit(1)
    print()

    try:
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

        if result.get("success"):
            print("\n" + "=" * 60)
            print("✓ DEPLOYMENT SUCCESSFUL")
            print("=" * 60)
            print(result.get("message", "Model deployed successfully!"))

            # Handle server process based on background mode
            if "server_process" in result and result["server_process"]:
                server_process = result["server_process"]
                
                if args.background or result.get("background"):
                    # Background mode: detach and exit
                    print(f"\n✓ Model server started in background on port {args.port}")
                    print(f"  PID: {server_process.pid}")
                    print(f"  To stop: kill {server_process.pid}")
                    # Detach from the process
                    sys.exit(0)
                else:
                    # Foreground mode: keep running
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
            print("✗ DEPLOYMENT FAILED")
            print("=" * 60)
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Need to import torch for the model loading
    import torch
    main()
