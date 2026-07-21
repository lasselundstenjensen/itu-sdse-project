"""
MLOps Pipeline Orchestration

Main pipeline script that orchestrates the complete ML workflow:
1. Data Fetching
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Registration
6. Deployment
"""

import sys

from .data.fetch import fetch_and_prepare_data
from .data.preprocess import preprocess_data
from .data.features import create_features
from .models.train import train_models
from .models.registry import register_models
from .deployment.deploy import deploy_model
from .utils import print_section_header


def run_data_pipeline(data: object = None) -> object:
    """
    Run the complete data pipeline.

    Parameters
    ----------
    data : pd.DataFrame, optional
        If provided, uses this data instead of fetching from file.

    Returns
    -------
    pd.DataFrame
        Fully processed data ready for model training.
    """
    print_section_header("DATA PIPELINE")

    # Fetch data
    if data is None:
        data = fetch_and_prepare_data()

    # Preprocess
    data = preprocess_data(data)

    # Feature engineering
    data = create_features(data)

    return data


def run_model_pipeline(data: object = None) -> dict:
    """
    Run the complete model pipeline.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Preprocessed data. If None, loads from file.

    Returns
    -------
    dict
        Dictionary with trained models and registration results.
    """
    print_section_header("MODEL PIPELINE")

    # Train models
    models_result = train_models(data)

    # Register models
    registry_result = register_models()

    # Deploy model if registered
    if registry_result.get("model_details"):
        model_version = registry_result["model_details"].get("version", 1)
        deploy_result = deploy_model(model_version=model_version)
        registry_result["deploy_success"] = deploy_result

    return {
        "models": models_result,
        "registry": registry_result,
    }


def run_full_pipeline() -> dict:
    """
    Run the complete end-to-end pipeline.

    Combines data and model pipelines into a single workflow.

    Returns
    -------
    dict
        Dictionary with results from both pipelines.
    """
    print_section_header("FULL MLOPS PIPELINE")
    print("Starting complete pipeline execution...\n")

    # Data pipeline
    data = run_data_pipeline()

    # Model pipeline
    model_results = run_model_pipeline(data)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print("\nSummary:")
    print(f"  Data shape: {data.shape}")
    print(f"  Models trained: {list(model_results.get('models', {}).keys())}")
    print(f"  Model registered: {model_results.get('registry', {}).get('model_details') is not None}")
    print(f"  Deployment successful: {model_results.get('registry', {}).get('deploy_success', False)}")

    return {
        "data": data,
        "models": model_results,
    }


if __name__ == "__main__":
    """
    Run the complete pipeline.

    Usage:
        python -m src.pipeline
        python src/pipeline.py
    """
    print("Running MLOps pipeline...")
    print("=" * 50)

    try:
        result = run_full_pipeline()
        print("\nPipeline executed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"\nPipeline failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
