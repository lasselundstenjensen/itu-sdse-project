from pathlib import Path
import pandas as pd
import json
import joblib
import shutil

from loguru import logger
from tqdm import tqdm
import typer
import mlflow

from mlops_pipeline.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from mlops_pipeline.modeling.xgboost_rf import train_xgboost_model
from mlops_pipeline.modeling.logreg import train_logreg_model
from mlops_pipeline.modeling.save_best_model import save_best_model
from mlops_pipeline.utils.save_artifacts import save_artifacts
from mlops_pipeline.data.io import load_X_y

# We have the MLFlow pipeline saved in mlflow_utils. Let's import:
from mlops_pipeline.utils.mlflow_utils import set_experiment, start_run, log_params, log_metrics, register_model, transition_stage, wait_until_ready

app = typer.Typer()


@app.command()
def train_models(
    # Paths to the four datasets
    X_train_path: Path = INTERIM_DATA_DIR / "X_train.csv",
    X_test_path: Path = INTERIM_DATA_DIR / "X_test.csv",
    y_train_path: Path = INTERIM_DATA_DIR / "y_train.csv",
    y_test_path: Path = INTERIM_DATA_DIR / "y_test.csv",

    # Paths to save the models
    xgboost_pkl_path: Path = MODELS_DIR / "lead_model_xgboost.pkl",
    xgboost_json_path: Path = MODELS_DIR / "lead_model_xgboost.json",
    logreg_pkl_path: Path = MODELS_DIR / "lead_model_logreg.pkl",



    # Trainer parameters
    random_state: int = 42,
    printing_bool: bool = False,

    # Let's add some mlflow parameters
    experiment_name: str = "Default",
    run_name: str = "xgboost_rf",
    register: bool = False,
    model_name: str = "LeadModel_XGBRF",
    stage: str = "Staging",
    data_version: str | None = None,
    #tags_kv: list[str] = typer.Option([], help="Additional tags, format: key=value")


):
    '''
    Train XGBoost RF using the pre-split csv's and saving a pkl and json of the best model.
    '''
    logger.info("Loading the split csv's...")
    X_train, y_train = load_X_y(X_train_path, y_train_path)
    X_test,  y_test  = load_X_y(X_test_path,  y_test_path)


    # Build tags dict from CLI - Here we can add tags for specific runs
    tags = {"model_type": "xgboost_rf"}
    # If this is a newer version of the model...
    if data_version:
        tags["data_version"] = data_version
    # Then we can merge additional key=value pairs
    #for kv in tags_kv:
    #    if "=" in kv:
    #        k, v = kv.split("=", 1)
    #        tags[k.strip()] = v.strip()

    #MLflow code below:
    set_experiment(experiment_name)

    with start_run(run_name=run_name, tags=tags) as run:
        logger.info(f"MLflow run_id: {run.info.run_id}")


        logger.info("Training XGBRFClassifier (randomized search)...")
        model_path, model_results, best_model = train_xgboost_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            xgboost_model_path=xgboost_pkl_path,
            xgboost_json_path=xgboost_json_path,
            random_state=random_state,
            printing=printing_bool,
        )


        mlflow.log_artifact(str(xgboost_pkl_path), artifact_path="model")
        mlflow.log_artifact(str(xgboost_json_path), artifact_path="model")

        # Convert nested report dict to a compact json artifact:
        report = next(iter(model_results.values())) # This is the dict
        report_path = MODELS_DIR / "classification_report_xgb_train.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path), artifact_path="metrics")


        # (Our trainer prints metrics, but doesn’t return test metrics; compute here)
        from sklearn.metrics import (
            accuracy_score, average_precision_score, roc_auc_score, classification_report
        )
        y_pred_test = best_model.predict(X_test)
        # XGBRFClassifier supports predict_proba for binary classification
        y_proba_test = best_model.predict_proba(X_test)[:, 1]

        metrics_test = {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "average_precision": float(average_precision_score(y_test, y_proba_test)),
            "roc_auc": float(roc_auc_score(y_test, y_proba_test)),
        }
        log_metrics(metrics_test)  # The helper from mlflow_utils


        # Also store the test classification report as an artifact
        report_test = classification_report(y_test, y_pred_test, output_dict=True)
        report_test_path = MODELS_DIR / "classification_report_xgb_test.json"
        report_test_path.write_text(json.dumps(report_test, indent=2))
        mlflow.log_artifact(str(report_test_path), artifact_path="metrics")

        
        logger.success(f"Saved scikit-learn wrapper to: {xgboost_pkl_path}")
        logger.success(f"Saved native XGBoost Booster JSON to: {xgboost_json_path}")

        # ==================================
        # Logistic Regression Training
        # ==================================
        logger.info("Training Logistic Regression (randomized search)...")

        model_path_lr, model_results_lr, best_model_lr = train_logreg_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            logreg_model_path=logreg_pkl_path,
            random_state=random_state,
            printing=printing_bool,
        )

        # Log model artifacts
        mlflow.log_artifact(str(logreg_pkl_path), artifact_path="model")

        # Train classification report (mirrors XGBoost logic)
        report_lr_train = next(iter(model_results_lr.values()))
        report_lr_train_path = MODELS_DIR / "classification_report_logreg_train.json"
        report_lr_train_path.write_text(
            json.dumps(report_lr_train, indent=2)
        )
        mlflow.log_artifact(
            str(report_lr_train_path),
            artifact_path="metrics",
        )

        # Test predictions
        y_pred_test = best_model_lr.predict(X_test)
        y_proba_test = best_model_lr.predict_proba(X_test)[:, 1]

        metrics_test = {
            "logreg_accuracy": float(
                accuracy_score(y_test, y_pred_test)
            ),
            "logreg_average_precision": float(
                average_precision_score(y_test, y_proba_test)
            ),
            "logreg_roc_auc": float(
                roc_auc_score(y_test, y_proba_test)
            ),
        }

        log_metrics(metrics_test)

        # Test classification report
        report_lr_test = classification_report(
            y_test, y_pred_test, output_dict=True
        )
        report_lr_test_path = MODELS_DIR / "classification_report_logreg_test.json"
        report_lr_test_path.write_text(
            json.dumps(report_lr_test, indent=2)
        )
        mlflow.log_artifact(
            str(report_lr_test_path),
            artifact_path="metrics",
        )

        # -------------------------------
        # Save best model to model.pkl
        # -------------------------------
        best_model_path, best_model_type = save_best_model(
            xgboost_model=best_model,
            lr_model=best_model_lr,
            xgboost_classification_report=report_test,
            lr_classification_report=report_lr_test,
            xgboost_model_path=xgboost_pkl_path,
            lr_model_path=logreg_pkl_path,
            printing=printing_bool
        )
        mlflow.log_artifact(str(best_model_path), artifact_path="model")
        logger.info(f"Best model ({best_model_type}) saved to: {best_model_path}")

        # ---- (Optional) Register model in MLflow Model Registry ----
        if register:
            # We registered the ARTIFACT path where we logged the model JSON under this run:
            #   runs:/<run_id>/model/lead_model_xgboost.json
            # Your helper expects the artifact_path relative to the run root:
            artifact_rel_path = f"model/{xgboost_json_path}"
            version = register_model(artifact_rel_path, model_name)
            logger.info(f"Requested registration: {model_name} v{version}")


            # Wait for registry to finish materializing the model
            ready = wait_until_ready(model_name, version, retries=20, sleep_seconds=1.0, raise_on_fail=True)
            if ready:
                transition_stage(model_name, version, stage=stage, archive_existing=True)
                logger.success(f"Model {model_name} v{version} transitioned to stage: {stage}")

    logger.success(f"Saved Logistic Regression model to: {logreg_pkl_path}")

    # We save artifacts based on the classification reports.
    save_artifacts(
        X_train, 
        MODELS_DIR / "classification_report_xgb_test.json",
        MODELS_DIR / "classification_report_logreg_test.json",
        MODELS_DIR / "artifacts" / "column_list.json",
        MODELS_DIR / "artifacts" / "model_results.json",
        False)
    logger.success(f"Artifacts have been saved to models/artifacts")



    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
