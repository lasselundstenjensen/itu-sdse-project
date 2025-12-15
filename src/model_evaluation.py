import json
import time
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

X_TEST_FILE = ARTIFACTS_DIR / "X_test.csv"
Y_TEST_FILE = ARTIFACTS_DIR / "y_test.csv"
LOGISTIC_REGRESSION_MODEL_FILE = ARTIFACTS_DIR / "logistic_regression_model.pkl"
XGBOOST_MODEL_FILE = ARTIFACTS_DIR / "xgboost_model.json"
FEATURE_COLUMN_LIST_FILE = ARTIFACTS_DIR / "feature_column_list.json"
MODEL_RESULTS_FILE = ARTIFACTS_DIR / "model_results.json"

def load_test_data():
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
    return X_test, y_test


def load_logistic_regression():
    return joblib.load(LOGISTIC_REGRESSION_MODEL_FILE)


def load_xgboost():
    return joblib.load(XGBOOST_MODEL_FILE)


def evaluate_predictions(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)


def save_columns_and_results(X, model_results, columns_file, results_file):
    FEATURE_COLUMN_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_COLUMN_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump({"columns": list(X.columns)}, f)

    with open(MODEL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2)


def wait_until_ready(model_name, version, max_retries=10, wait_sec=1):
    client = MlflowClient()
    for _ in range(10):
        details = client.get_model_version(
            name=model_name,
            version=version,
        )
        status = ModelVersionStatus.from_string(details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            return
        time.sleep(1)


def register_model():
    client = MlflowClient()
    model_uri = mlflow.get_artifact_uri(
        artifact_path=artifact_path,
        run_id=run_id,
    )
    details = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
    wait_until_ready(details.name, details.version)
    return dict(details)


def evaluation_pipeline():
    X_test, y_test = load_test_data()

    lr_model = load_logistic_regression()
    xgb_model = load_xgboost()

    y_pred_lr = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    lr_report = evaluate_predictions(y_test, y_pred_lr)
    xgb_report = evaluate_predictions(y_test, y_pred_xgb)

    model_results = {
        "logistic_regression": lr_report,
        "xgboost": xgb_report,
    }

    save_columns_and_results(X_test, model_results, FEATURE_COLUMN_LIST_FILE, MODEL_RESULTS_FILE)


    print("Evaluation complete.")
    print(f"- {FEATURE_COLUMN_LIST_FILE}")
    print(f"- {MODEL_RESULTS_FILE}")


if __name__ == "__main__":
    evaluation_pipeline()

#do we have a master script? (i guess main?) and i guess we should have it but we should check