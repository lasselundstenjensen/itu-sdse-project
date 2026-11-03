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
IMAGE_DATA_DIR = RAW_DATA_DIR / "images"
METADATA_PATH = IMAGE_DATA_DIR / "metadata.csv"
TEST_DATA_DIR = RAW_DATA_DIR / "test"

TRAINING_CONFIG_PATH = PROJ_ROOT / "vials" / "modeling" / "training_config.yaml"

MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PYTORCH_MODEL_PATH = MODELS_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACTS_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACTS_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACTS_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACTS_DIR / "inference_log.json"

# MLflow - we log to a local sqlite db so no tracking server has to be running
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_TRACKING_URI = f"sqlite:///{(MLFLOW_STATE_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_LOCATION = (MLFLOW_STATE_DIR / "artifacts").as_uri()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
