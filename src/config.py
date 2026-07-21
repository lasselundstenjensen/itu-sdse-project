"""
Configuration for the MLOps Pipeline

Centralized constants and paths used throughout the pipeline.
All values can be modified directly in this file.
"""

import datetime
from pathlib import Path

# =============================================================================
# DATE CONFIGURATION
# =============================================================================

# Default date range for training data
# These can be overridden when calling the fetch functions
MIN_DATE = "2024-01-01"
MAX_DATE = "2024-01-31"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory for artifacts (relative to project root)
ARTIFACT_DIR = Path("./artifacts")

# MLflow tracking directory
MLRUNS_DIR = Path("./mlruns")

# Data paths
# Note: raw_data.csv is currently in notebooks/artifacts/ from the original setup
RAW_DATA_PATH = Path("./notebooks/artifacts/raw_data.csv")
TRAIN_DATA_GOLD_PATH = ARTIFACT_DIR / "train_data_gold.csv"
TRAIN_DATA_PATH = ARTIFACT_DIR / "training_data.csv"

# Scaler and preprocessor paths
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"

# Model paths
XGBOOST_MODEL_PATH = ARTIFACT_DIR / "lead_model_xgboost.json"
LR_MODEL_PATH = ARTIFACT_DIR / "lead_model_lr.pkl"

# Metadata paths
COLUMNS_DRIFT_PATH = ARTIFACT_DIR / "columns_drift.json"
COLUMNS_LIST_PATH = ARTIFACT_DIR / "columns_list.json"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DATE_LIMITS_PATH = ARTIFACT_DIR / "date_limits.json"
CAT_MISSING_IMPUTE_PATH = ARTIFACT_DIR / "cat_missing_impute.csv"
OUTLIER_SUMMARY_PATH = ARTIFACT_DIR / "outlier_summary.csv"

# Test data paths (from existing notebooks directory)
X_TEST_PATH = Path("./notebooks/artifacts/X_test.csv")
Y_TEST_PATH = Path("./notebooks/artifacts/y_test.csv")

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

# Experiment name - uses current date by default
# This creates a new experiment each day
EXPERIMENT_NAME = datetime.datetime.now().strftime("%Y_%B_%d")

# Model registry name
MODEL_NAME = "lead_model"

# Artifact path for MLflow model logging
ARTIFACT_PATH = "model"

# Data version for tracking
DATA_VERSION = "00000"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Test size for train/test split
TEST_SIZE = 0.15

# Outlier handling - z-score threshold
OUTLIER_Z_THRESHOLD = 2.0

# Imputation method for numeric columns
IMPUTATION_METHOD = "mean"  # or "median"


if __name__ == "__main__":
    """Print configuration when run directly."""
    print("MLOps Pipeline Configuration")
    print("=" * 50)
    print(f"Date Range: {MIN_DATE} to {MAX_DATE}")
    print(f"Artifact Directory: {ARTIFACT_DIR.resolve()}")
    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Test Size: {TEST_SIZE}")
