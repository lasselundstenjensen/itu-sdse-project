"""
Configuration for the Image Classification MLOps Pipeline

Centralized constants and paths used throughout the pipeline for glass vial
image classification using PyTorch CNN.
All values can be modified directly in this file.
"""

import datetime
import os
from pathlib import Path

# =============================================================================
# DATE CONFIGURATION
# =============================================================================

# Default date range for training data (kept for compatibility)
MIN_DATE = "2024-01-01"
MAX_DATE = "2024-01-31"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory for artifacts (relative to project root)
ARTIFACT_DIR = Path("./artifacts")

# MLflow tracking directory
MLRUNS_DIR = Path("./mlruns")

# Image data paths
IMAGE_DATA_DIR = Path("./data/images/raw")
METADATA_PATH = Path("./data/images/metadata.csv")

# Model paths
PYTORCH_MODEL_PATH = ARTIFACT_DIR / "vial_cnn_model.pth"

# Metadata paths for image classification
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR / "image_statistics.json"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

# Experiment name for image classification
EXPERIMENT_NAME = "vial_image_classification"

# Model registry name
MODEL_NAME = "cnn_model"

# Artifact path for MLflow model logging
ARTIFACT_PATH = "model"

# Data version for tracking
DATA_VERSION = "00000"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# =============================================================================
# IMAGE CLASSIFICATION CONFIGURATION
# =============================================================================

# Image specifications
IMAGE_SIZE = (64, 64)  # Balanced between quality and training speed for 1600x768 source images
ORIGINAL_IMAGE_SIZE = (1600, 768)  # Native resolution of ICPR 2024 dataset
NUM_CLASSES = 2

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 0.0001

# Transform parameters (ImageNet statistics)
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

# Classification threshold (tunable)
THRESHOLD = 0.5

# Device configuration (auto-detected) - import lazily to avoid dependency issues
_DEVICE = None

def get_device():
    """Get the appropriate device for PyTorch operations."""
    global _DEVICE
    if _DEVICE is None:
        try:
            import torch
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            # Fallback if PyTorch is not available
            _DEVICE = "cpu"
    return _DEVICE

# For backward compatibility, use the function or import torch when needed
DEVICE = None  # Will be set to actual device when torch is available

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# DRIFT DETECTION CONFIGURATION
# =============================================================================

# Drift detection window size (number of samples to collect before checking)
DRIFT_WINDOW_SIZE = 100

# Population Stability Index (PSI) threshold for drift detection
DRIFT_PSI_THRESHOLD = 0.2

# Kolmogorov-Smirnov test p-value threshold for drift detection
DRIFT_KS_PVALUE_THRESHOLD = 0.05

# Interval between drift checks (in seconds)
DRIFT_CHECK_INTERVAL = 60

# Path to save/load baseline statistics for drift detection
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR / "drift_baseline_stats.json"

# Path to save/load reference data for drift detection
DRIFT_REFERENCE_DATA_PATH = ARTIFACT_DIR / "drift_reference_data.pkl"

# Feature layer to extract embeddings from for drift detection
DRIFT_FEATURE_LAYER = "fc1"

# File to store the monitoring run ID
MONITORING_RUN_ID_FILE = ARTIFACT_DIR / "monitoring_run_id.txt"

# =============================================================================
# INFERENCE LOGGING CONFIGURATION
# =============================================================================

# Path to the inference request log file
INFERENCE_LOG_PATH = ARTIFACT_DIR / "inference_log.json"

# Default time window for drift checking (in days)
DRIFT_DEFAULT_DAYS = 7

# =============================================================================
# MLFLOW SERVER CONFIGURATION
# =============================================================================

# MLflow tracking URI - local server (overridable via MLFLOW_TRACKING_URI env var)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


if __name__ == "__main__":
    """Print configuration when run directly."""
    print("Image Classification MLOps Pipeline Configuration")
    print("=" * 50)
    print(f"Date Range: {MIN_DATE} to {MAX_DATE}")
    print(f"Artifact Directory: {ARTIFACT_DIR.resolve()}")
    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    device = get_device()
    print(f"Device: {device}")
    print(f"Classification Threshold: {THRESHOLD}")
    print(f"\nDrift Detection Configuration:")
    print(f"  Window Size: {DRIFT_WINDOW_SIZE}")
    print(f"  PSI Threshold: {DRIFT_PSI_THRESHOLD}")
    print(f"  KS P-value Threshold: {DRIFT_KS_PVALUE_THRESHOLD}")
    print(f"  Check Interval: {DRIFT_CHECK_INTERVAL}s")
    print(f"  Feature Layer: {DRIFT_FEATURE_LAYER}")
    print(f"\nMLflow Configuration:")
    print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"\nInference Logging Configuration:")
    print(f"  Log Path: {INFERENCE_LOG_PATH}")
    print(f"  Drift Default Days: {DRIFT_DEFAULT_DAYS}")
