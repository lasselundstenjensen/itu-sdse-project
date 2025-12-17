# mlops_project/scripts/main.py

import datetime
import os
import pickle
import json
import numpy as np
import sys
import traceback
from mlops_project.data.load import pull_data, load_raw_data
from mlops_project.data.preprocess import preprocess
from mlops_project.features.build_features import build_features
from mlops_project.models.train import train_models
from mlops_project.models.evaluate import evaluate_model, compare_models


def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def main():
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting data pipeline...")
        pull_data()
        data = load_raw_data()
        data = preprocess(data)
        data = build_features(data)
        
        print("Training models...")
        experiment_name = datetime.datetime.now().strftime("%Y_%B_%d")
        xgb_model, lr_model, X_test, y_test = train_models(data, experiment_name)
        
        print("Evaluating models...")
        xgb_results = evaluate_model(xgb_model.best_estimator_, X_test, y_test)
        lr_results = evaluate_model(lr_model.best_estimator_, X_test, y_test)
        
        results = {
            "xgboost": xgb_results,
            "logistic_regression": lr_results
        }
        
        best = compare_models(results)
        print(f"Pipeline completed! Best model: {best}")
        
        print("Saving artifacts...")
        
        # Save best model with standard name 'model' for GitHub workflow
        best_model = lr_model.best_estimator_ if best == "logistic_regression" else xgb_model.best_estimator_
        with open(f"{output_dir}/model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print(f"✓ Saved best model: {best}")
        
        # Save both models
        with open(f"{output_dir}/xgboost_model.pkl", "wb") as f:
            pickle.dump(xgb_model.best_estimator_, f)
        
        with open(f"{output_dir}/logistic_regression_model.pkl", "wb") as f:
            pickle.dump(lr_model.best_estimator_, f)
        print("✓ Saved all models")
        
        # Save results (convert NumPy types)
        results_serializable = convert_to_serializable(results)
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results_serializable, f, indent=2)
        print("✓ Saved results")
        
        # Save best model info
        with open(f"{output_dir}/best_model.txt", "w") as f:
            f.write(f"Best model: {best}\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"F1 Score: {results[best]['report']['weighted avg']['f1-score']:.4f}\n")
            f.write(f"Accuracy: {results[best]['accuracy']:.4f}\n")
        print("✓ Saved model metadata")
        
        print(f"\n✓ All artifacts saved to {output_dir}/")
        print(f"✓ Model artifact ready for GitHub workflow: {output_dir}/model")
        
        # Ensure we exit cleanly
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR in main pipeline:")
        print(f"❌ {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())