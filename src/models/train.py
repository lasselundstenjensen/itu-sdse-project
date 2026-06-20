"""
Model Training Module

Handles training of machine learning models including XGBoost and Logistic Regression.
Includes hyperparameter tuning with RandomizedSearchCV and MLflow integration.
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.stats import randint, uniform
from xgboost import XGBRFClassifier

from ..config import (
    ARTIFACT_DIR,
    COLUMNS_LIST_PATH,
    EXPERIMENT_NAME,
    LR_MODEL_PATH,
    MODEL_RESULTS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_DATA_GOLD_PATH,
    XGBOOST_MODEL_PATH,
)
from ..data.features import bin_categorical_columns
from ..utils import create_dummy_cols, print_section_header


def setup_mlflow() -> str:
    """
    Setup MLflow experiment and directories.

    Creates necessary directories and sets the active experiment.

    Returns
    -------
    str
        The experiment name that was set.
    """
    print(f"Setting up MLflow experiment: {EXPERIMENT_NAME}")

    # Ensure directories exist
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    Path("./mlruns").mkdir(parents=True, exist_ok=True)
    Path("./mlruns/.trash").mkdir(parents=True, exist_ok=True)

    # Set experiment
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        print(f"Warning: Could not set MLflow experiment: {e}")

    print(f"MLflow experiment set to: {EXPERIMENT_NAME}")
    return EXPERIMENT_NAME


def load_training_data() -> pd.DataFrame:
    """
    Load preprocessed training data from the gold dataset.

    Returns
    -------
    pd.DataFrame
        Training data ready for model preparation.

    Raises
    ------
    FileNotFoundError
        If training data file does not exist.
    """
    print(f"Loading training data from {TRAIN_DATA_GOLD_PATH}")

    if not TRAIN_DATA_GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA_GOLD_PATH}. "
            "Run data pipeline first."
        )

    data = pd.read_csv(TRAIN_DATA_GOLD_PATH)
    print(f"Training data loaded. Shape: {data.shape}")
    return data


def prepare_data_for_training(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for model training by creating dummy variables.

    Drops ID columns, identifies categorical columns, creates one-hot encoding,
    and converts all columns to float64.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data with both categorical and continuous columns.

    Returns
    -------
    pd.DataFrame
        Data ready for model training (all float64, with dummy variables).
    """
    print("\nPreparing data for training...")

    # Drop columns not needed for training
    drop_cols = ["lead_id", "customer_code", "date_part"]
    data = data.drop([col for col in drop_cols if col in data.columns], axis=1)

    # Identify categorical columns
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_cols = [col for col in cat_cols if col in data.columns]

    if cat_cols:
        cat_vars = data[cat_cols]
        other_vars = data.drop(cat_cols, axis=1)

        # Create dummy variables
        print("Creating dummy variables for categorical columns...")
        for col in cat_vars.columns:
            cat_vars[col] = cat_vars[col].astype("category")
            cat_vars = create_dummy_cols(cat_vars, col)
            print(f"  Created dummies for: {col}")

        data = pd.concat([other_vars, cat_vars], axis=1)
    else:
        other_vars = data

    # Convert all to float64
    for col in data.columns:
        data[col] = data[col].astype("float64")

    print(f"Data prepared for training. Shape: {data.shape}")
    return data


def split_data(
    data: pd.DataFrame,
    target_col: str = "lead_indicator",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.

    Uses stratified splitting to maintain class distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Data with features and target.
    target_col : str, default="lead_indicator"
        Name of the target column.
    test_size : float, default=TEST_SIZE from config
        Proportion of data to use for testing.
    random_state : int, default=RANDOM_STATE from config
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        - X_train: Training features
        - X_test: Test features
        - y_train: Training targets
        - y_test: Test targets
    """
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    y = data[target_col]
    X = data.drop([target_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=random_state,
        test_size=test_size,
        stratify=y,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    cv: int = 10,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """
    Train XGBoost model with randomized hyperparameter search.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    n_iter : int, default=10
        Number of parameter settings sampled.
    cv : int, default=10
        Number of cross-validation folds.
    random_state : int, default=RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    RandomizedSearchCV
        Fitted XGBoost model with best parameters.
    """
    print("\n" + "=" * 50)
    print("TRAINING XGBOOST MODEL")
    print("=" * 50)

    model = XGBRFClassifier(random_state=random_state, n_estimators=100)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }

    model_grid = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_jobs=-1,
        verbose=3,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
    )

    print("Starting XGBoost training with RandomizedSearchCV...")
    model_grid.fit(X_train, y_train)
    print("XGBoost training complete!")

    return model_grid


def evaluate_and_save_xgboost(
    model_grid: RandomizedSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBRFClassifier:
    """
    Evaluate XGBoost model and save the best estimator.

    Parameters
    ----------
    model_grid : RandomizedSearchCV
        Fitted XGBoost model grid.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.

    Returns
    -------
    XGBRFClassifier
        Best XGBoost estimator.
    """
    print("\n" + "=" * 50)
    print("XGBOOST EVALUATION")
    print("=" * 50)

    # Get best parameters
    best_params = model_grid.best_params_
    print("\nBest XGBoost parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Predictions
    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_pred_train, y_train)
    test_acc = accuracy_score(y_pred_test, y_test)
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

    # Classification reports
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))

    print("\nTrain Classification Report:")
    print(classification_report(y_train, y_pred_train))

    # Save best model
    best_model = model_grid.best_estimator_
    XGBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model.save_model(XGBOOST_MODEL_PATH)
    print(f"\nBest XGBoost model saved to {XGBOOST_MODEL_PATH}")

    return best_model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[LogisticRegression, RandomizedSearchCV]:
    """
    Train Logistic Regression model with MLflow logging.

    Uses RandomizedSearchCV for hyperparameter tuning and logs
    metrics and artifacts to MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.
    n_iter : int, default=10
        Number of parameter settings sampled.
    cv : int, default=3
        Number of cross-validation folds.
    random_state : int, default=RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    tuple[LogisticRegression, RandomizedSearchCV]
        - Best Logistic Regression estimator
        - Fitted RandomizedSearchCV object
    """
    print("\n" + "=" * 50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 50)

    # Define custom MLflow wrapper for probability prediction
    class LRWrapper(mlflow.pyfunc.PythonModel):
        """Wrapper to predict probabilities instead of classes."""

        def __init__(self, model):
            self.model = model

        def predict(self, context, model_input):
            """Predict probability of positive class."""
            return self.model.predict_proba(model_input)[:, 1]

    # Setup MLflow autologging
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)

    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else None

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression(max_iter=1000, random_state=random_state)

        params = {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "penalty": ["none", "l1", "l2", "elasticnet"],
            "C": [100, 10, 1.0, 0.1, 0.01],
        }

        model_grid = RandomizedSearchCV(
            model,
            param_distributions=params,
            verbose=3,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )

        print("Starting Logistic Regression training...")
        model_grid.fit(X_train, y_train)
        print("Logistic Regression training complete!")

        best_model = model_grid.best_estimator_

        # Predictions
        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        # Log metrics
        f1 = f1_score(y_test, y_pred_test)
        mlflow.log_metric("f1_score", f1)

        # Log artifacts
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="model")
        mlflow.log_param("data_version", "00000")

        # Save model
        LR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=best_model, filename=LR_MODEL_PATH)
        print(f"Logistic Regression model saved to {LR_MODEL_PATH}")

        # Log custom model for probability prediction
        mlflow.pyfunc.log_model("model", python_model=LRWrapper(best_model))

    # Print best parameters
    best_params = model_grid.best_params_
    print("\nBest Logistic Regression parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Print accuracy
    train_acc = accuracy_score(y_pred_train, y_train)
    test_acc = accuracy_score(y_pred_test, y_test)
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

    # Classification reports
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))

    print("\nTrain Classification Report:")
    print(classification_report(y_train, y_pred_train))

    return best_model, model_grid


def save_model_artifacts(
    X_train: pd.DataFrame,
    model_results: dict,
) -> None:
    """
    Save model artifacts including column list and results.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (used to get column names).
    model_results : dict
        Dictionary mapping model paths to their evaluation results.
    """
    print("\nSaving model artifacts...")

    # Save column list
    COLUMNS_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = {"column_names": list(X_train.columns)}
    with open(COLUMNS_LIST_PATH, "w") as columns_file:
        json.dump(columns, columns_file, indent=2)
    print(f"Column list saved to {COLUMNS_LIST_PATH}")

    # Save model results
    MODEL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_RESULTS_PATH, "w") as results_file:
        json.dump(model_results, results_file, indent=2)
    print(f"Model results saved to {MODEL_RESULTS_PATH}")


def train_models(
    data: pd.DataFrame = None,
) -> dict:
    """
    Complete model training pipeline.

    If data is not provided, loads from TRAIN_DATA_GOLD_PATH.

    Trains both XGBoost and Logistic Regression models, evaluates them,
    and saves all artifacts.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Preprocessed training data. If None, loads from file.

    Returns
    -------
    dict
        Dictionary with trained models and their evaluation results.
    """
    print_section_header("MODEL TRAINING")

    # Setup MLflow
    setup_mlflow()

    # Load data if not provided
    if data is None:
        data = load_training_data()

    # Prepare data
    data = prepare_data_for_training(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Train XGBoost
    xgboost_model_grid = train_xgboost(X_train, y_train)
    xgboost_model = evaluate_and_save_xgboost(
        xgboost_model_grid, X_train, y_train, X_test, y_test
    )

    # Train Logistic Regression
    lr_model, lr_model_grid = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )

    # Build model results
    model_results = {
        str(XGBOOST_MODEL_PATH): classification_report(
            y_train, xgboost_model_grid.predict(X_train), output_dict=True
        ),
        str(LR_MODEL_PATH): classification_report(
            y_test, lr_model_grid.predict(X_test), output_dict=True
        ),
    }

    # Save artifacts
    save_model_artifacts(X_train, model_results)

    return {
        "xgboost": xgboost_model,
        "logistic_regression": lr_model,
        "results": model_results,
    }


if __name__ == "__main__":
    """
    Run model training module directly.

    Usage:
        python -m src.models.train
    """
    print("Running model training module...")
    try:
        models = train_models()
        print("\nModel training complete!")
        print(f"Trained models: {list(models.keys())}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
