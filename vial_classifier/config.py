"""Project-wide configuration: paths, data source, training and serving settings.

Single source of truth for the constants that used to live in the configuration
cells of notebooks/01_training.ipynb and notebooks/02_deployment.ipynb.
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# --- Paths --------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Raw images: metadata.csv files are DVC-tracked, the JPEGs are downloaded next to them
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
# Training set (730 images, split 70/15/15 into train/val/test at training time)
TRAIN_DATA_DIR = RAW_IMAGES_DIR / "train"
TRAIN_METADATA_PATH = TRAIN_DATA_DIR / "metadata.csv"
# Held-out test set (10 images, 2 per class), never used for training
TEST_DATA_DIR = RAW_IMAGES_DIR / "test"
TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.csv"

MODELS_DIR = PROJ_ROOT / "models"
# Exported MLflow pyfunc model - uploaded as the `model` CI artifact and baked into the image
EXPORT_DIR = MODELS_DIR / "model"

ARTIFACT_DIR = PROJ_ROOT / "artifacts"
PYTORCH_MODEL_PATH = ARTIFACT_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR / "inference_log.json"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Data source (public Hetzner object store) ---------------------------------
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"
S3_TRAIN_PREFIX = "images/raw/"
S3_TEST_PREFIX = "test/"

# --- Training -----------------------------------------------------------------
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

# --- Images -------------------------------------------------------------------
IMAGE_SIZE = (64, 64)
ORIGINAL_IMAGE_SIZE = (1600, 768)
NUM_CLASSES = 2
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
# Kept exactly as in the training notebook (note: "good", not "clean")
CLASS_NAMES = {0: "good", 1: "defective"}

# --- MLflow (local SQLite store, no tracking server) ----------------------------
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"
MLFLOW_TRACKING_URI = f"sqlite:///{(MLFLOW_STATE_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_LOCATION = (MLFLOW_STATE_DIR / "artifacts").as_posix()
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MODEL_ARTIFACT_PATH = "model_pyfunc"

# --- Serving & drift ----------------------------------------------------------
SERVING_PORT = 5001
UPSTREAM_PORT = 5002
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"
UPSTREAM_HOST = "127.0.0.1"
DRIFT_DEFAULT_DAYS = 7
DRIFT_KS_PVALUE_THRESHOLD = 0.05

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
