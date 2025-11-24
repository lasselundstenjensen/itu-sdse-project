"""Main pipeline orchestrator for the ML workflow."""

from loguru import logger
from pathlib import Path
import joblib
import mlflow
from pathlib import Path
from loguru import logger

from itu_sdse_project.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from itu_sdse_project.modeling.loaders import load_raw_data, pull_dvc_data


def train():
    """
    Complete training pipeline:
    - Load data 
    - Process data
    - Train models
    - Evaluate and select best model
    - Register model
    """
    logger.info("=" * 50)
    logger.info("Starting TRAINING pipeline...")
    logger.info("=" * 50)
    
    # 1. Load data
    logger.info("Step 1: Loading data...")
    pull_dvc_data()
    df = load_raw_data(RAW_DATA_DIR / "raw_data.csv")
    
    # 2. Clean data
    logger.info("Step 2: Cleaning data...")
    # Cleaning goes here
    df_cleaned = df  # Placeholder for cleaned data
    
    # 3. Transform features
    logger.info("Step 3: Transforming features...")
    # transforming goes
    df_transformed = df_cleaned  # Placeholder for transformed data
    
    # 4. Split data
    logger.info("Step 4: Splitting train/test...")
    # X_train, X_test, y_train, y_test = split_train_test(df_transformed)
    
    # Save processed data
    #X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    #X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    #y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    #y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
    
    # 5. Train models
    
    # 6. Evaluate
    
    # 7. Register best model

"""Prediction and model loading utilities."""

def load_scaler(scaler_path: Path):
    """Load saved scaler from disk."""
    logger.info(f"Loading scaler from {scaler_path}")
    # TODO: Use joblib.load() to load scaler
    pass


def load_model(model_name: str = "lead_model", stage: str = "Production"):
    """Load model from MLflow registry."""
    logger.info(f"Loading model: {model_name} (stage: {stage})")
    # TODO: Use mlflow.pyfunc.load_model() or load from file
    pass


def load_model_from_file(model_path: Path):
    """Load model from saved file."""
    logger.info(f"Loading model from {model_path}")
    # TODO: Load XGBoost or pickle model from file
    pass


if __name__ == "__main__":
    train()