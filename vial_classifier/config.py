from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"

# Training configuration
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42

# Image specifications
IMAGE_SIZE = (64, 64)
ORIGINAL_IMAGE_SIZE = (1600, 768)
NUM_CLASSES = 2
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

# Paths (relative to the project root)
ARTIFACT_DIR = "artifacts"
MLFLOW_STATE_DIR = "mlflow-state"
IMAGE_DATA_DIR = "./data/images/raw"
METADATA_PATH = "./data/images/metadata.csv"
TEST_DATA_DIR = "./data/test"
PYTORCH_MODEL_PATH = ARTIFACT_DIR + "/vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR + "/model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR + "/drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR + "/class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR + "/image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR + "/inference_log.json"
DRIFT_DEFAULT_DAYS = 7
DRIFT_KS_PVALUE_THRESHOLD = 0.05

# MLflow - we log to a local sqlite db so no tracking server has to be running
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = "./mlflow-state/artifacts"
