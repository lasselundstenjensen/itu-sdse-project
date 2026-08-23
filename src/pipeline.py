import sys
from torch.utils.data import DataLoader

from .data.fetch import fetch_and_prepare_data
from .data.preprocess import preprocess_data
from .data.features import create_features
from .models.train import train_models
from .models.registry import register_models
from .deployment.deploy import deploy_model, transition_to_production, set_model_alias
from .utils import print_section_header


def run_data_pipeline(data=None):
    print_section_header("IMAGE DATA PIPELINE")

    data = fetch_and_prepare_data()

    train_loader, val_loader, test_loader = preprocess_data(data)
    create_features()

    return train_loader, val_loader, test_loader


def run_model_pipeline(train_loader=None, val_loader=None, test_loader=None, deploy_to_production=False):
    print_section_header("CNN MODEL PIPELINE")

    models_result = train_models(train_loader, val_loader, test_loader)
    registry_result = register_models()

    if registry_result.get("model_details"):
        model_version = registry_result["model_details"].get("version", 1)
        deploy_result = deploy_model(
            model_version=model_version,
            transition_to_prod=deploy_to_production
        )
        registry_result["deploy_success"] = deploy_result

        if deploy_result:
            set_model_alias(model_version=model_version, alias="best")
            registry_result["model_labelled"] = True

            if deploy_to_production:
                prod_result = transition_to_production(model_version=model_version)
                registry_result["production_deploy_success"] = prod_result

    return {
        "models": models_result,
        "registry": registry_result,
    }


def run_full_pipeline(deploy_to_production=False):
    print_section_header("FULL IMAGE CLASSIFICATION PIPELINE")
    print("Starting complete pipeline execution...\n")

    train_loader, val_loader, test_loader = run_data_pipeline()

    model_results = run_model_pipeline(
        train_loader, val_loader, test_loader,
        deploy_to_production=deploy_to_production
    )

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

    result = run_full_pipeline(deploy_to_production=args.deploy_to_prod)
    print("\nPipeline executed successfully!")
