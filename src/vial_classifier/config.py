"""Central configuration: hyperparameters, paths and remote locations.

Values are copied from the configuration cells of the original notebooks so
that the refactored pipeline trains and serves exactly the same model.
All relative paths are resolved against the project root, so the code works
no matter which directory it is started from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Remote data (public Hetzner object store)
# ---------------------------------------------------------------------------
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"
S3_DOWNLOAD_WORKERS = 8

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "images" / "raw"
IMAGE_METADATA_PATH = DATA_DIR / "images" / "metadata.csv"
TEST_DATA_DIR = DATA_DIR / "test"
TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_CHECKPOINT_PATH = ARTIFACT_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR / "inference_log.json"

# Exported, servable model directory (this is what is baked into the image)
EXPORT_MODEL_DIR = PROJECT_ROOT / "models" / "model"

MLFLOW_STATE_DIR = PROJECT_ROOT / "mlflow-state"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_STATE_DIR / "artifacts")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
NUM_CLASSES = 2
CLASS_NAMES = {0: "good", 1: "defective"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = (64, 64)
ORIGINAL_IMAGE_SIZE = (1600, 768)
# ImageNet statistics, as used by the original notebook
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3

# ---------------------------------------------------------------------------
# MLflow registry
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
ARTIFACT_PATH = "model_pyfunc"
TRAINING_RUN_NAME = "training_notebook"

# ---------------------------------------------------------------------------
# Serving and drift
# ---------------------------------------------------------------------------
API_PORT = 5001  # port the FastAPI front end listens on (matches app.yaml)
SCORING_PORT = 5002  # internal port of the MLflow scoring subprocess
SCORING_HOST = "127.0.0.1"
DRIFT_DEFAULT_DAYS = 7
DRIFT_KS_PVALUE_THRESHOLD = 0.05
DRIFT_MEAN_THRESHOLD = 0.1
