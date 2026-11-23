"""Settings shared by all pipeline steps (paths, data split, training).

Everything here comes from the configuration cell of notebooks/01_training.ipynb,
so the .py steps use exactly the same values as the original notebook.
"""

from pathlib import Path

# ---------------------------------------------------------------- Paths
# Project root = the folder that contains src/ (works no matter where you run from)
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "images" / "raw"
METADATA_PATH = DATA_DIR / "images" / "metadata.csv"
TEST_DATA_DIR = DATA_DIR / "test"
TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
PYTORCH_MODEL_PATH = ARTIFACTS_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACTS_DIR / "model_results.json"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACTS_DIR / "image_statistics.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACTS_DIR / "drift_baseline_stats.json"
INFERENCE_LOG_PATH = ARTIFACTS_DIR / "inference_log.json"

# ---------------------------------------------------------------- Object store
# Public Hetzner bucket with the images (no credentials needed)
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"

# ---------------------------------------------------------------- Images
IMAGE_SIZE = (64, 64)  # images are resized from 1600x768 to this
ORIGINAL_IMAGE_SIZE = (1600, 768)
NUM_CLASSES = 2
CLASS_NAMES = {0: "good", 1: "defective"}
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ---------------------------------------------------------------- Training
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3

# ---------------------------------------------------------------- MLflow
# Logs to a local sqlite db, so no tracking server has to be running
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_STATE_DIR / "artifacts")
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
