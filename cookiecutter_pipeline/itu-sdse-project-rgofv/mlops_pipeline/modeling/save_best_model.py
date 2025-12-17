from pathlib import Path
import joblib
import shutil
from loguru import logger
from mlops_pipeline.config import MODELS_DIR

def save_best_model(
    xgboost_model,
    lr_model,
    xgboost_classification_report: dict,
    lr_classification_report: dict,
    xgboost_model_path: Path = None,
    lr_model_path: Path = None,
    output_path: Path = None,
    printing: bool = False
) -> tuple[Path, str]:
    """
    Evaluate and save the best performing model (XGBoost vs Logistic Regression) as model.pkl.
    Compares models based on weighted average f1-score from classification reports.
    
    Args:
        xgboost_model: Trained XGBoost model object
        lr_model: Trained Logistic Regression model object
        xgboost_classification_report: Classification report dict for XGBoost model
        lr_classification_report: Classification report dict for Logistic Regression model
        xgboost_model_path: Optional. Path to XGBoost model file (for copying if needed)
        lr_model_path: Optional. Path to Logistic Regression model file (for copying if needed)
        output_path: Optional. Path where to save the best model. Defaults to MODELS_DIR / "model.pkl"
        printing: Whether to print status messages
    
    Returns:
        tuple: (Path to saved model file, model type name)
    """
    if output_path is None:
        output_path = MODELS_DIR / "model.pkl"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract f1-scores from classification reports
    xgb_f1_score = xgboost_classification_report.get("weighted avg", {}).get("f1-score", 0.0)
    lr_f1_score = lr_classification_report.get("weighted avg", {}).get("f1-score", 0.0)
    
    if printing:
        logger.info(f"XGBoost weighted avg f1-score: {xgb_f1_score:.4f}")
        logger.info(f"Logistic Regression weighted avg f1-score: {lr_f1_score:.4f}")
    
    # Compare and save the better model
    if xgb_f1_score >= lr_f1_score:
        # XGBoost is better or equal
        joblib.dump(xgboost_model, output_path)
        model_type = "XGBoost"
        if printing:
            logger.info(f"Best model: XGBoost (f1-score: {xgb_f1_score:.4f} >= {lr_f1_score:.4f})")
            logger.info(f"Saved best model (XGBoost) to {output_path}")
    else:
        # Logistic Regression is better
        if lr_model_path and lr_model_path.exists():
            # Copy the existing pickle file if path is provided
            shutil.copy(lr_model_path, output_path)
        else:
            # Otherwise dump the model object
            joblib.dump(lr_model, output_path)
        model_type = "Logistic Regression"
        if printing:
            logger.info(f"Best model: Logistic Regression (f1-score: {lr_f1_score:.4f} > {xgb_f1_score:.4f})")
            logger.info(f"Saved best model (Logistic Regression) to {output_path}")
    
    return output_path, model_type