import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
import os
import time
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
import mlflow
from mlflow.tracking.client import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import base64
from datetime import datetime, timezone
from io import BytesIO
import mlflow.pyfunc
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path
from pprint import pprint
import warnings
import atexit
import subprocess
import sys
import uvicorn
from fastapi import FastAPI, Query, Request, Response
from scipy.stats import ks_2samp
import urllib.error
import urllib.request
import threading
import requests

warnings.filterwarnings('ignore')


<<<<<<< HEAD
from read_load_data import get_image_dataset
from preprocessing import define_transforms, split_dataset, create_dataloaders
from utils import calculate_image_statistics, print_section_header
from train import SimpleCNN, train_cnn_model, evaluate_cnn_model, tune_threshold, compute_baseline_statistics, SimpleLoggingWrapper
from model_registry import wait_until_ready, register_model
=======
from src.data.read_load_data import get_image_dataset
from src.data.preprocessing import define_transforms, split_dataset, create_dataloaders
from src.utils import calculate_image_statistics, print_section_header
from src.model.train import SimpleCNN, train_cnn_model, evaluate_cnn_model, tune_threshold, compute_baseline_statistics, SimpleLoggingWrapper
from src.model.model_registry import wait_until_ready, register_model
from src.model.model_selection import setup_mlflow_tracking, find_best_run, get_production_model, wait_for_deployment, promote_model, export_serving_model
from src.deployment.serve_model_locally import start_scoring_server
from src.deployment.drift_detection import load_inference_logs, filter_logs_by_time, load_baseline_stats, calculate_confidence_metrics, check_confidence_drift, check_class_distribution_drift, detect_drift
from src.deployment.fastapi import _upstream_health
from tests.test_endpoints import test_endpoint
from tests.test_drift import test_drift
>>>>>>> 7a04966 (added the deployment part to main)

max_date = "2024-01-31"
min_date = "2024-01-01"

# Training configuration
num_epochs = 5
batch_size = 32
learning_rate = 0.0003
threshold = 0.5
early_stop_patience = 2
scheduler_patience = 3
random_state = 42

# Image specifications
image_size = (64, 64)
original_image_size = (1600, 768)
num_classes = 2
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]

# Paths (relative to the project root)
#PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ".../itu-mlops-26-our-group"
artifact_dir = "artifacts"
mlflow_state_dir = "mlflow-state"
image_data_dir = "PROJECT_ROOT/data/images/raw"
metadata_path = "PROJECT_ROOT/data/images/metadata.csv"
test_data_dir = "PROJECT_ROOT/data/test"
pytorch_model_path = artifact_dir + "/vial_cnn_model.pth"
model_results_path = artifact_dir + "/model_results.json"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
drift_default_days = 7
drift_ks_pvalue_threshold = 0.05
class_names_path = artifact_dir + "/class_names.json"
image_statistics_path = artifact_dir + "/image_statistics.json"
inference_log_path = artifact_dir + "/inference_log.json"

# MLflow - we log to a local sqlite db so no tracking server has to be running
experiment_name = "vial_image_classification"
model_name = "cnn_model"
artifact_path = "model_pyfunc"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"

# Where the exported servable model goes (this is what gets baked into the Docker image)
export_dir = "models/model"
model_path = export_dir

# Serving ports
port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

# Where the endpoint lives in production
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"

#experiment = mlflow.get_experiment_by_name(experiment_name)
#experiment_ids = [experiment.experiment_id]

setup_mlflow_tracking(experiment_name)
# raw image dataset
image_dataset = get_image_dataset()
# define transformations
train_transforms, val_transforms = define_transforms(image_size, transform_mean, transform_std)
# split data
train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)
# 
train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size, train_transforms=train_transforms, val_transforms=val_transforms)


# Save the image statistics of the training set as an artifact
image_stats = calculate_image_statistics(train_loader)
with open(image_statistics_path, "w") as f:
    json.dump(image_stats, f, indent=2)
print(f"Image statistics saved to {image_statistics_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print_section_header("CNN MODEL TRAINING")

torch.manual_seed(random_state)
np.random.seed(random_state)

if mlflow.active_run():
    mlflow.end_run()
    print("Ended previous active MLflow run")

run = mlflow.start_run(run_name="training_notebook")

# Log parameters to MLflow
mlflow.log_param("model_architecture", "SimpleCNN")
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("num_epochs", num_epochs)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("initial_threshold", threshold)
mlflow.log_param("num_classes", num_classes)
mlflow.log_param("image_size", "64x64")
mlflow.log_param("random_seed", random_state)

# Log model architecture details
model_params = sum(p.numel() for p in SimpleCNN().parameters())
trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
mlflow.log_param("total_parameters", model_params)
mlflow.log_param("trainable_parameters", trainable_params)

cnn_model, training_history = train_cnn_model(
    train_loader, val_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device,
    random_seed=random_state,
)

# Tune threshold on validation set
optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)

# Evaluate on test set with optimal threshold
test_results = evaluate_cnn_model(cnn_model, test_loader, device=device, threshold=optimal_threshold)

#baseline
baseline_stats = compute_baseline_statistics(cnn_model, val_loader, threshold=optimal_threshold)

with open(drift_baseline_stats_path, 'w') as f:
    json.dump(baseline_stats, f, indent=2)
print(f"Baseline statistics saved to {drift_baseline_stats_path}")

# Log baseline statistics to MLflow
mlflow.log_metric("baseline_image_mean_r", baseline_stats['image_statistics']['mean_r'])
mlflow.log_metric("baseline_image_mean_g", baseline_stats['image_statistics']['mean_g'])
mlflow.log_metric("baseline_image_mean_b", baseline_stats['image_statistics']['mean_b'])
mlflow.log_metric("baseline_image_std_r", baseline_stats['image_statistics']['std_r'])
mlflow.log_metric("baseline_image_std_g", baseline_stats['image_statistics']['std_g'])
mlflow.log_metric("baseline_image_std_b", baseline_stats['image_statistics']['std_b'])
mlflow.log_metric("baseline_class_0_count", baseline_stats['class_distribution']['class_0_count'])
mlflow.log_metric("baseline_class_1_count", baseline_stats['class_distribution']['class_1_count'])
mlflow.log_metric("baseline_confidence_mean", baseline_stats['confidence_statistics']['mean'])
mlflow.log_metric("baseline_confidence_std", baseline_stats['confidence_statistics']['std'])
mlflow.log_artifact(drift_baseline_stats_path, "drift_detection")

#logging wrapper defined, save model artifacts
print("\nSaving model artifacts...")

# Save model state dict
torch.save({
    'model_state_dict': cnn_model.state_dict(),
    'architecture': SimpleCNN.__name__,
    'input_shape': (3, 64, 64),
    'output_shape': (1,),
    'threshold': optimal_threshold,
    'num_classes': num_classes,
}, pytorch_model_path)
print(f"CNN model saved to {pytorch_model_path}")

# Save model results
with open(model_results_path, "w") as results_file:
    json.dump(test_results, results_file, indent=2)
print(f"Model results saved to {model_results_path}")

mlflow.log_param("classification_threshold", optimal_threshold)

# Log the pyfunc model with the SimpleLoggingWrapper as the model artifact
wrapper = SimpleLoggingWrapper(model=cnn_model)
mlflow.pyfunc.log_model("model_pyfunc", python_model=wrapper)
print("Logged pyfunc model to the MLflow run")

#model registration
model_uri = f"runs:/{run.info.run_id}/model_pyfunc"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
wait_until_ready(model_details.name, model_details.version)

print(f"Model registered successfully:")
print(f"  Name: {model_details.name}")
print(f"  Version: {model_details.version}")
print(f"  Stage: {model_details.current_stage}")

# Transition to staging
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_details.version,
    stage="Staging",
    archive_existing_versions=True,
)
client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
client.set_model_version_tag(model_name, model_details.version, "task", "classification")
client.set_registered_model_alias(name=model_name, alias="best", version=str(model_details.version))

print(f"Successfully transitioned {model_name} version {model_details.version} to Staging")
pprint(dict(model_details))

mlflow.end_run()

print("\n" + "=" * 50)
print("TRAINING JOB COMPLETE")
print("=" * 50)
print("\nSummary:")
print(f"  Training samples: {len(train_loader.dataset)}")
print(f"  Validation samples: {len(val_loader.dataset)}")
print(f"  Test samples: {len(test_loader.dataset)}")
print(f"  Optimal threshold: {optimal_threshold:.4f}")
print(f"  Test F1-score: {test_results['f1_score']:.4f}")
print(f"  Registered model: {model_name} version {model_details.version} (Staging)")

#deployemnt
setup_mlflow_tracking(experiment_name)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

#model selection
experiment_best, run_id, score = find_best_run(experiment_name)

with open(model_results_path, "r") as f:
    model_results = json.load(f)
pprint(model_results["classification_report"])

production_model_uri, prod_model_version, prod_model_run_id = get_production_model(client, model_name)

train_model_score = experiment_best.get("metrics.test_f1_score")
if train_model_score is None:
    train_model_score = experiment_best.get("metrics.f1_score")

model_status = {}
#run_id = None

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

print("run_id:", run_id)
print("artifact_path:", artifact_path)

if run_id is None:
    raise ValueError(
        "run_id is None. Check the return value of find_best_run()."
    )

version=promote_model(client, model_name, run_id, artifact_path)

export_serving_model(client, model_name, experiment_name, artifact_path, export_dir)

scoring_proc = start_scoring_server()
print("Scoring server starting on port", upstream_port)

#starting fastapi
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


#test endpoint
test_endpoint()
print("GET /version ->", requests.get(f"http://localhost:{port}/version").json())
print("GET /health  ->", requests.get(f"http://localhost:{port}/health").json())

test_drift(drift_default_days)

api_server.should_exit = True
if scoring_proc is not None and scoring_proc.poll() is None:
    scoring_proc.terminate()
    scoring_proc.wait(timeout=10)
    print("Scoring server stopped")
print("API server stopped")