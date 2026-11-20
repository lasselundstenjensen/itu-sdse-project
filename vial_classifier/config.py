from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
IMAGE_DATA_DIR = DATA_DIR / "images" / "raw"
METADATA_PATH = DATA_DIR / "images" / "metadata.csv"
TEST_DATA_DIR = DATA_DIR / "test"
TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.csv"

MODELS_DIR = PROJ_ROOT / "models"
MODEL_EXPORT_DIR = MODELS_DIR / "model"

ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
PYTORCH_MODEL_PATH = ARTIFACTS_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACTS_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACTS_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACTS_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACTS_DIR / "inference_log.json"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"
MLFLOW_ARTIFACTS_DIR = MLFLOW_STATE_DIR / "artifacts"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_ARTIFACTS_DIR)

MIN_DATE = "2024-01-01"
MAX_DATE = "2024-01-31"

NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42

IMAGE_SIZE = (64, 64)
ORIGINAL_IMAGE_SIZE = (1600, 768)
NUM_CLASSES = 2

TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {
    0: "good",
    1: "defective",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"
MAX_DOWNLOAD_WORKERS = 8

EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_MODEL_ARTIFACT_PATH = "model_pyfunc"
MODEL_VERSION_FALLBACK = 1

DRIFT_DEFAULT_DAYS = 7
DRIFT_KS_PVALUE_THRESHOLD = 0.05
DRIFT_MEAN_THRESHOLD = 0.1

API_HOST = "0.0.0.0"
API_PORT = 5001
UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 5002

PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

if tqdm is not None:
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)