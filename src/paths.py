from pathlib import Path

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILE = DATA_DIR / "raw_data.csv"

DATE_FILTERED_DATA_FILE = ARTIFACTS_DIR / "date_filtered_data.csv"
TRAINING_DATA_FILE = ARTIFACTS_DIR / "training_data.csv"
TRAIN_GOLD_FILE = ARTIFACTS_DIR / "train_data_gold.csv"

X_TEST_FILE = ARTIFACTS_DIR / "X_test.csv"
Y_TEST_FILE = ARTIFACTS_DIR / "y_test.csv"

DATE_METADATA_FILE = ARTIFACTS_DIR / "date_limits.json"
DATE_LIMITS_FILE = ARTIFACTS_DIR / "date_limits.json"
OUTLIER_SUMMARY_FILE = ARTIFACTS_DIR / "outlier_summary.csv"
CAT_MISSING_IMPUTE_FILE = ARTIFACTS_DIR / "cat_missing_impute.csv"
SCALER_FILE = ARTIFACTS_DIR / "scaler.pkl"
COLUMNS_DRIFT_FILE = ARTIFACTS_DIR / "columns_drift.json"
COLUMNS_LIST_FILE = ARTIFACTS_DIR / "columns_list.json"
MODEL_RESULTS_FILE = ARTIFACTS_DIR / "model_results.json"

XGBOOST_MODEL_FILE = ARTIFACTS_DIR / "lead_model_xgboost.json"
LR_MODEL_FILE = ARTIFACTS_DIR / "lead_model_lr.pkl"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLRUNS_TRASH_DIR = MLRUNS_DIR / ".trash"
