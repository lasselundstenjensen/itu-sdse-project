"""
One-off export script to move latest MLflow run artifacts to data/output/

Usage:
    python -m src.scripts.export_mlflow_run
"""

import os
from pathlib import Path
import mlflow
import shutil


def main():
    """Export latest MLflow run artifacts to data/output/ directory."""
    # Enable file store for MLflow
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    
    # Set experiment
    experiment_name = "vial_image_classification"
    mlflow.set_experiment(experiment_name)
    
    # Find latest run in this experiment
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        max_results=1,
        order_by=["start_time DESC"]
    )
    
    if runs.empty:
        raise RuntimeError(f"No MLflow runs found for experiment '{experiment_name}'")
    
    run_id = runs.iloc[0].run_id
    print(f"Found latest run: {run_id}")
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model artifact
    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model",
            dst_path=str(output_dir)
        )
        print(f"Downloaded model artifacts to: {artifact_path}")
        
        # MLflow downloads to a 'model' subdirectory, extract the actual model.pth
        model_file = Path(artifact_path) / "data" / "model.pth"
        if model_file.exists():
            shutil.move(str(model_file), str(output_dir / "model.pth"))
            print(f"Moved model to: {output_dir / 'model.pth'}")
            # Clean up the downloaded directory
            shutil.rmtree(Path(artifact_path))
        else:
            print(f"Warning: Expected model file not found at {model_file}")
    except Exception as e:
        print(f"Warning: Could not download model artifacts: {e}")
        # Try to find the model in the artifacts directory
        artifact_dir = Path("./artifacts")
        model_path = artifact_dir / "vial_cnn_model.pth"
        if model_path.exists():
            print(f"Using local model from: {model_path}")
            shutil.copy(model_path, output_dir / "model.pth")
        else:
            raise RuntimeError("No model found in MLflow or local artifacts")
    
    # Get threshold from params
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)
    threshold = run_data.data.params.get("optimal_threshold", 0.5)
    
    # Save threshold
    threshold_path = output_dir / "threshold.txt"
    threshold_path.write_text(str(threshold))
    print(f"Saved threshold: {threshold} to {threshold_path}")
    
    print(f"\nExported run {run_id} to {output_dir}")
    print(f"Contents of {output_dir}:")
    for item in output_dir.iterdir():
        print(f"  - {item.name}")


if __name__ == "__main__":
    main()
