from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env file if it exists
load_dotenv()

# Standard cookiecutter paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Project data (layout kept from the original repo because DVC tracks it there)
IMAGE_DIR = DATA_DIR / "images" / "raw"
METADATA_PATH = DATA_DIR / "images" / "metadata.csv"
TEST_DATA_DIR = DATA_DIR / "test"
TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.csv"

# Pipeline outputs
ARTIFACT_DIR = PROJ_ROOT / "artifacts"
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"
PYTORCH_MODEL_PATH = ARTIFACT_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR / "inference_log.json"

# Object store (public Hetzner bucket)
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"

# Training
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Images
IMAGE_SIZE = (64, 64)
ORIGINAL_IMAGE_SIZE = (1600, 768)
NUM_CLASSES = 2
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "good", 1: "defective"}

# MLflow (local SQLite, no tracking server needed)
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_STATE_DIR / "artifacts")
