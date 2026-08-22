"""
Image Classification MLOps Pipeline Orchestration

Main pipeline script that orchestrates the complete ML workflow for glass vial image classification:
1. Image Data Fetching
2. Image Data Preprocessing (PyTorch DataLoaders)
3. Feature Engineering (Image artifacts)
4. CNN Model Training
5. Model Registration
6. Deployment
"""

import sys
from typing import Tuple
from pathlib import Path
from torch.utils.data import DataLoader

from .data.fetch import fetch_and_prepare_data
from .data.preprocess import preprocess_data
from .data.features import create_features
from .models.train import train_models
from .models.registry import register_models
from .deployment.deploy import deploy_model, transition_to_production, set_model_alias
from .utils import print_section_header


def run_data_pipeline(data: object = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Run the complete image data pipeline.

    Parameters
    ----------
    data : List[Tuple[Path, int]], optional
        If provided, uses this dataset instead of fetching from files.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        - Training DataLoader
        - Validation DataLoader
        - Test DataLoader
    """
    print_section_header("IMAGE DATA PIPELINE")

    # Fetch and prepare image data
    if data is None:
        data = fetch_and_prepare_data()

    # Preprocess data (creates DataLoaders)
    train_loader, val_loader, test_loader = preprocess_data(data)

    # Create and save feature artifacts
    create_features()

    return train_loader, val_loader, test_loader


def run_model_pipeline(
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    deploy_to_production: bool = False,
) -> dict:
    """
    Run the complete CNN model pipeline.

    Parameters
    ----------
    train_loader : DataLoader, optional
        Training DataLoader. If None, loads from data pipeline.
    val_loader : DataLoader, optional
        Validation DataLoader. If None, loads from data pipeline.
    test_loader : DataLoader, optional
        Test DataLoader. If None, loads from data pipeline.
    deploy_to_production : bool, default=False
        If True, deploy the best model directly to Production after Staging.

    Returns
    -------
    dict
        Dictionary with trained CNN model and registration results.
    """
    print_section_header("CNN MODEL PIPELINE")

    # Train CNN model
    models_result = train_models(train_loader, val_loader, test_loader)

    # Register model
    registry_result = register_models()

    # Deploy model if registered
    if registry_result.get("model_details"):
        model_version = registry_result["model_details"].get("version", 1)
        deploy_result = deploy_model(
            model_version=model_version,
            transition_to_prod=deploy_to_production
        )
        registry_result["deploy_success"] = deploy_result
        
        # Label the best model
        if deploy_result:
            set_model_alias(model_version=model_version, alias="best")
            registry_result["model_labelled"] = True
            
            # Optionally transition to Production
            if deploy_to_production:
                prod_result = transition_to_production(model_version=model_version)
                registry_result["production_deploy_success"] = prod_result

    return {
        "models": models_result,
        "registry": registry_result,
    }


def run_full_pipeline(deploy_to_production: bool = False) -> dict:
    """
    Run the complete end-to-end image classification pipeline.

    Combines data and model pipelines into a single workflow.

    Parameters
    ----------
    deploy_to_production : bool, default=False
        If True, deploy the best model directly to Production after Staging.

    Returns
    -------
    dict
        Dictionary with results from both pipelines.
    """
    print_section_header("FULL IMAGE CLASSIFICATION PIPELINE")
    print("Starting complete pipeline execution...\n")

    # Data pipeline
    train_loader, val_loader, test_loader = run_data_pipeline()

    # Model pipeline
    model_results = run_model_pipeline(
        train_loader, val_loader, test_loader,
        deploy_to_production=deploy_to_production
    )

    # Get some info about the data
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print("\nSummary:")
    print(f"  Training samples: {train_samples}")
    print(f"  Validation samples: {val_samples}")
    print(f"  Test samples: {test_samples}")
    print(f"  CNN model trained: {model_results.get('models', {}).get('cnn_model') is not None}")
    print(f"  Model registered: {model_results.get('registry', {}).get('model_details') is not None}")
    print(f"  Deployment successful: {model_results.get('registry', {}).get('deploy_success', False)}")
    print(f"  Model labelled as best: {model_results.get('registry', {}).get('model_labelled', False)}")
    print(f"  Production deployment: {model_results.get('registry', {}).get('production_deploy_success', False)}")
    if model_results.get('models', {}).get('optimal_threshold'):
        print(f"  Optimal threshold: {model_results['models']['optimal_threshold']:.4f}")
    if model_results.get('models', {}).get('test_results'):
        test_f1 = model_results['models']['test_results'].get('f1_score', 0)
        print(f"  Test F1-score: {test_f1:.4f}")

    return {
        "data": {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        },
        "models": model_results,
    }


if __name__ == "__main__":
    """
    Run the complete pipeline.

    Usage:
        python -m src.pipeline
        python src/pipeline.py
        python src/pipeline.py --deploy-to-prod  # Deploy directly to Production
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the complete image classification MLOps pipeline"
    )
    parser.add_argument(
        '--deploy-to-prod',
        action='store_true',
        help="Deploy the best model directly to Production after Staging",
    )
    
    args = parser.parse_args()
    
    print("Running MLOps pipeline...")
    print("=" * 50)

    try:
        result = run_full_pipeline(deploy_to_production=args.deploy_to_prod)
        print("\nPipeline executed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"\nPipeline failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
