import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
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


from src.data.read_load_data import get_image_dataset
from src.data.preprocessing import define_transforms, split_dataset, create_dataloaders
from utils import calculate_image_statistics, print_section_header
from src.model.train import SimpleCNN, train_cnn_model, evaluate_cnn_model, tune_threshold, compute_baseline_statistics, SimpleLoggingWrapper
from src.model.model_registry import wait_until_ready, register_model

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
artifact_dir = "artifacts"
mlflow_state_dir = "mlflow-state"
image_data_dir = "./data/images/raw"
metadata_path = "./data/images/metadata.csv"
test_data_dir = "./data/test"
pytorch_model_path = artifact_dir + "/vial_cnn_model.pth"
model_results_path = artifact_dir + "/model_results.json"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
class_names_path = artifact_dir + "/class_names.json"
image_statistics_path = artifact_dir + "/image_statistics.json"
inference_log_path = artifact_dir + "/inference_log.json"

# MLflow - we log to a local sqlite db so no tracking server has to be running
experiment_name = "vial_image_classification"
model_name = "cnn_model"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"

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