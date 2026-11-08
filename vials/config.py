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
TEST_DATA_DIR = RAW_DATA_DIR / "test"

MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_RESULTS_PATH = ARTIFACTS_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACTS_DIR / "drift_baseline_stats.json"
INFERENCE_LOG_PATH = ARTIFACTS_DIR / "inference_log.json"

EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_TRACKING_URI = f"sqlite:///{(MLFLOW_STATE_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_LOCATION = (MLFLOW_STATE_DIR / "artifacts").as_uri()

# For serving and drift detection
ARTIFACT_PATH = "model_pyfunc"  # name of the model inside an MLflow run

DRIFT_DEFAULT_DAYS = 7
DRIFT_KS_PVALUE_THRESHOLD = 0.05
DRIFT_MEAN_DIFF_THRESHOLD = 0.1

# Where the exported servable model goes (this is what gets baked into the Docker image)
MODEL_PATH = MODELS_DIR / "model"

# Serving ports
PORT = 5001
UPSTREAM_PORT = 5002
UPSTREAM_HOST = "127.0.0.1"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
